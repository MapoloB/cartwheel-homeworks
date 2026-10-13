"""Homework 1: the remaining commerce-agent tools.

The three lecture tools (`search_help_center`, `get_order`, `issue_refund`)
are implemented in agent/agent.py and are worked examples of the pattern:
check permissions first, go through agent/db.py for data, and return a
structured dict, never a prose error. The homework tools follow the same
pattern. agent/agent.py already wraps each function below as an SDK tool, so
once a function works here it works in chat with no further wiring.

Result convention (see agent/auth.py):
  - Success: a dict with "ok": True plus the payload fields named in each
    docstring.
  - Failure: {"ok": False, "error": <code>, "reason": <human-readable str>}.

Run the contract tests with: uv run pytest tests/test_hw_holes.py -k hw1
They are marked xfail and flip to passing as you implement each function.
"""

from __future__ import annotations

import difflib
import re
from datetime import timedelta
from typing import Any

from rapidfuzz import fuzz

from agent import db
from agent.auth import AuthContext, can_cancel_order, can_view_order, permission_denied
from agent.config import load_facts
from agent.helpcenter import load_policy_docs
from agent.killswitch import kill_switch
from seed.eligibility import effective_return_window_days, is_refund_eligible, refund_needs_approval

MAX_SEARCH_LIMIT = 25
DEFAULT_ORDER_LIMIT = 20


def get_policy(ctx: AuthContext, policy_id: str) -> dict[str, Any]:
    """Fetch one policy doc by its exact id. Risk tier: read.

    Every role may read every policy doc (the corpus is public help-center
    content), so this tool needs no permission check.

    Args:
        ctx: The caller's auth context. Unused here, but every tool takes it.
        policy_id: An exact policy id, e.g. "cw-returns" or
            "store-juniper-home-goods-policy". Matching is exact and
            case-sensitive; ids are the `policy_id` front-matter field of the
            files in data/policies/.

    Returns:
        On success: {"ok": True, "policy_id": str, "title": str,
        "audience": str, "body": str} where body is the markdown body of the
        doc without the front matter.
        If no doc has that id: {"ok": False, "error": "not_found",
        "reason": ...} naming the id that was requested.

    Implementation notes:
        agent.helpcenter.load_policy_docs() returns every parsed doc.
    """
    for doc in load_policy_docs():
        if doc.policy_id == policy_id:
            return {
                "ok": True,
                "policy_id": doc.policy_id,
                "title": doc.title,
                "audience": doc.audience,
                "body": doc.body,
            }
    return {
        "ok": False,
        "error": "not_found",
        "reason": (
            f"No policy doc has id {policy_id!r}. Ids are exact and case-sensitive; "
            "use search_help_center to find the right id."
        ),
    }


def search_products(
    ctx: AuthContext,
    query: str,
    store: str | None = None,
    max_price_usd: float | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search the product catalog. Risk tier: read.

    Every role may search products. Matching is deterministic keyword
    matching, not semantic search: a product matches when every whitespace
    token of `query` appears case-insensitively as a substring of the
    product's title or description.

    Args:
        ctx: The caller's auth context.
        query: Free-text query. Must be non-empty after stripping whitespace;
            otherwise return {"ok": False, "error": "invalid_argument",
            "reason": ...}.
        store: Optional store filter. Matched with
            agent.db.get_store_by_name (case-insensitive name or slug). If
            given and no store matches, return {"ok": False, "error":
            "not_found", "reason": ...} naming the store string.
        max_price_usd: Optional inclusive price ceiling. If given and not
            strictly positive, return an "invalid_argument" error.
        limit: Maximum products to return. Clamp to the range
            [1, MAX_SEARCH_LIMIT]; do not error on out-of-range values.

    Returns:
        {"ok": True, "products": [...], "count": <len(products)>} where each
        product is {"product_id": int, "store_id": int, "title": str,
        "price_usd": float}. Sort matches by price_usd ascending, then by
        product_id ascending, and truncate to `limit`. No matches is still a
        success: {"ok": True, "products": [], "count": 0}.

    Implementation notes:
        agent.db.list_products(conn, store_id) gives the candidate set.
        Use `with db.connection() as conn:` to close the database automatically.
    """
    tokens = query.lower().split()
    if not tokens:
        return {"ok": False, "error": "invalid_argument", "reason": "query must be non-empty."}
    if max_price_usd is not None and max_price_usd <= 0:
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": f"max_price_usd must be positive, got {max_price_usd}.",
        }
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))

    with db.connection() as conn:
        store_id = None
        if store is not None:
            found = db.get_store_by_name(conn, store)
            if found is None:
                return {
                    "ok": False,
                    "error": "not_found",
                    "reason": _unknown_store_reason(conn, store),
                }
            store_id = found.id
        candidates = db.list_products(conn, store_id)

    products = []
    for p in candidates:
        text = f"{p.title} {p.description}".lower()
        if not all(token in text for token in tokens):
            continue
        price_usd = p.price_cents / 100
        if max_price_usd is not None and price_usd > max_price_usd:
            continue
        products.append(
            {"product_id": p.id, "store_id": p.store_id, "title": p.title, "price_usd": price_usd}
        )
    products.sort(key=lambda p: (p["price_usd"], p["product_id"]))
    products = products[:limit]
    return {"ok": True, "products": products, "count": len(products)}


def _unknown_store_reason(conn: Any, store: str) -> str:
    """Name the unmatched store and suggest close store names, without guessing for the model."""
    names = [row[0] for row in conn.execute("SELECT name FROM stores ORDER BY name")]
    wanted = store.lower()
    suggestions = [n for n in names if wanted in n.lower()]
    lowered = {n.lower(): n for n in names}
    for close in difflib.get_close_matches(wanted, list(lowered), n=3, cutoff=0.6):
        if lowered[close] not in suggestions:
            suggestions.append(lowered[close])
    reason = f"No store named {store!r}."
    if suggestions:
        reason += " Did you mean: " + ", ".join(suggestions[:3]) + "?"
    return reason


def list_my_orders(ctx: AuthContext) -> dict[str, Any]:
    """List recent orders in the caller's own scope. Risk tier: read.

    Role behavior, straight from the access matrix in SPEC.md:
        - shopper: the caller's own orders.
        - merchant: the caller's store's orders (ctx.store_id).
        - support: support staff have no orders of their own and look up
          specific orders with get_order instead, so return {"ok": False,
          "error": "invalid_argument", "reason": ...} saying exactly that.

    Returns:
        For shopper and merchant: {"ok": True, "orders": [...],
        "count": <len(orders)>} where each order is
        agent.db.Order.to_public_dict() and the list holds at most
        DEFAULT_ORDER_LIMIT orders, newest first (agent.db.list_orders_for_user
        and list_orders_for_store already sort and limit this way).
        "has_more" is True when older orders exist beyond the returned list,
        so tell the user these are only their most recent orders.

    Implementation notes:
        No permission check is needed beyond the role dispatch, because the
        scope is baked into which query you run. That is the point of the
        tool: the model cannot ask for someone else's orders through it.
    """
    if ctx.role == "support":
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": (
                "Support staff have no orders of their own; look up a specific "
                "order with get_order instead."
            ),
        }
    if ctx.role == "merchant" and ctx.store_id is None:
        return permission_denied("Merchant account has no store on file.")
    if ctx.role not in {"shopper", "merchant"}:
        return permission_denied(f"Unknown role {ctx.role!r}.")

    # Fetch one extra row so the model can tell the shopper the list is truncated.
    with db.connection() as conn:
        if ctx.role == "shopper":
            rows = db.list_orders_for_user(conn, ctx.user_id, limit=DEFAULT_ORDER_LIMIT + 1)
        else:
            rows = db.list_orders_for_store(conn, ctx.store_id, limit=DEFAULT_ORDER_LIMIT + 1)
    orders = [o.to_public_dict() for o in rows[:DEFAULT_ORDER_LIMIT]]
    return {
        "ok": True,
        "orders": orders,
        "count": len(orders),
        "has_more": len(rows) > DEFAULT_ORDER_LIMIT,
    }


def cancel_order(ctx: AuthContext, order_id: int, reason: str) -> dict[str, Any]:
    """Cancel an order. Risk tier: write.

    This is the homework's write tool, and it must enforce two independent
    rules in this order:

    1. The access matrix (scope): use agent.auth.can_cancel_order. Shoppers
       may cancel only their own orders, merchants only their own store's
       orders, support any order. On failure return
       agent.auth.permission_denied(...) with a reason naming the role and
       the order id. Scope is checked before the status rule so that an
       out-of-scope caller learns nothing about the order's state.
    2. The pre-shipment rule (facts.yaml `cancel_cutoff`): only orders whose
       status is exactly "placed" can be cancelled, for every role. If the
       order is in scope but its status is not "placed", return
       {"ok": False, "error": "not_eligible", "reason": ...} that names the
       current status and states that orders can be cancelled only before
       shipment.

    Args:
        ctx: The caller's auth context.
        order_id: The order to cancel.
        reason: Free-text reason from the user; not validated.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found",
        "reason": ...}.
        On success: {"ok": True, "order_id": order_id, "status": "cancelled"}
        after persisting the new status with agent.db.set_order_status.

    Implementation notes:
        Fetch with agent.db.get_order. Note the argument order of
        can_cancel_order(ctx, order_user_id, order_store_id).

    The Module 4 kill switch is checked first (before the scope and
    status rules and before your code), so that a paused write tool touches
    nothing. It is provided; the default ("off") returns None and falls
    through to your implementation.
    """
    paused = kill_switch("cancel_order")
    if paused is not None:
        return {"ok": False, "error": "paused", "reason": paused}
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"No order has id {order_id}."}
        if not can_cancel_order(ctx, order.user_id, order.store_id):
            return permission_denied(f"A {ctx.role} may not cancel order {order_id}.")
        if order.status != "placed":
            reason = (
                f"Order {order_id} is {order.status!r}; orders can be cancelled "
                "only before shipment."
            )
            if order.status in {"shipped", "delivered"}:
                reason += " A return or refund may still be possible; see policy cw-returns."
            return {"ok": False, "error": "not_eligible", "reason": reason}
        db.set_order_status(conn, order_id, "cancelled")
    return {"ok": True, "order_id": order_id, "status": "cancelled"}


def find_order(ctx: AuthContext, query: str) -> dict[str, Any]:
    """Search the caller's orders by product name. Risk tier: read.

    Takes a natural-language query (e.g., "earmuffs I bought last week")
    and searches the authenticated user's orders for products whose name
    matches. Use fuzzy string matching (e.g., thefuzz.fuzz.partial_ratio
    or case-insensitive substring matching) to find orders whose product name is close to the
    query.

    Access rules: a shopper searches only the shopper's own orders, a
    merchant searches orders from the merchant's store, and support staff
    can search any orders. Use agent.db.list_order_search_candidates with
    user_id=ctx.user_id for shoppers, store_id=ctx.store_id for merchants,
    or all_orders=True only for support. Derive the scope from ctx, never
    from the query; reject unsupported roles or missing required identity.
    Use agent.db.list_products to map product IDs to product titles.

    The helper returns the complete authorised scope, newest first with
    order ID descending as the tie-breaker. Match product names first,
    preserve that order, then return at most five matches. Do not search
    only the 20 most recent orders. Convert matches with to_public_dict().

    Args:
        ctx: The caller's auth context.
        query: A natural-language description of the product.

    Returns:
        {"ok": True, "orders": [...]} with a list of matching orders
        (at most 5), each as the dict returned by agent.db. If no orders
        match, return {"ok": True, "orders": []}. Matching is strict: every
        descriptive word must match the product name. If several orders
        match, confirm with the user which one they mean before acting on it.
    """
    if ctx.role == "shopper":
        scope: dict[str, Any] = {"user_id": ctx.user_id}
    elif ctx.role == "merchant" and ctx.store_id is not None:
        scope = {"store_id": ctx.store_id}
    elif ctx.role == "support":
        scope = {"all_orders": True}
    else:
        return permission_denied(f"A {ctx.role} without a valid scope may not search orders.")

    words = [w for w in _words(query) if w not in _QUERY_FILLER]
    if not words:
        return {"ok": True, "orders": []}

    with db.connection() as conn:
        titles = {p.id: p.title for p in db.list_products(conn)}
        candidates = db.list_order_search_candidates(conn, **scope)

    matching_products = {
        pid for pid, title in titles.items() if _title_matches(words, _words(title))
    }
    matches = [o for o in candidates if o.product_id in matching_products]
    return {"ok": True, "orders": [o.to_public_dict() for o in matches[:5]]}


# Words people use to describe a purchase that never name the product itself.
_QUERY_FILLER = {
    "a", "an", "the", "i", "me", "my", "mine", "we", "our", "you", "your", "it", "that", "this",
    "those", "these", "one", "some", "of", "for", "from", "to", "in", "on", "at", "with", "and",
    "or", "is", "was", "were", "be", "been", "did", "do", "have", "had", "got", "get", "bought",
    "buy", "ordered", "order", "orders", "purchased", "purchase", "item", "items", "thing",
    "last", "past", "recent", "recently", "ago", "earlier", "yesterday", "today", "week",
    "weeks", "month", "months", "year", "days", "day", "where", "what", "which", "find",
    "show", "look", "up", "please", "can", "could", "help",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _title_matches(query_words: list[str], title_words: list[str]) -> bool:
    """Strict match: every query word must closely match some title word.

    A false match can lead to refunding or cancelling the wrong order, so
    this prefers missing an order over returning a different product.
    """
    return bool(title_words) and all(
        any(fuzz.ratio(q, t) >= 85 for t in title_words) for q in query_words
    )


def check_refund_eligibility(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """Explain whether one order is refund-eligible, and why. Risk tier: read.

    Student-added tool (HW1 Part A). Part B showed the agent guessing why an
    order was ineligible ("likely due to store policies or the time since
    delivery") without a citation. This tool returns the facts behind the
    decision so the agent can explain and cite it.

    Access: the same view check as get_order; an out-of-scope caller learns
    nothing about the order.

    Returns:
        {"ok": True, "order_id", "eligible", "status", "delivered_at",
        "days_since_delivery", "window_days", "window_source" ("platform" or
        "store_override"), "deadline", "policy_id", "total_usd",
        "needs_approval", "reason", "data_issue"}.
        `eligible` is the order's stamped refund_eligible flag, which is what
        issue_refund enforces, so the two tools never disagree. `data_issue`
        is None, or a description of an inconsistent record; when it is set,
        the agent should escalate instead of stating a deadline.
        Errors: "not_found" for an unknown order, "permission_denied" outside
        the caller's scope.
    """
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"No order has id {order_id}."}
        if not can_view_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not view order #{order_id}"
            )
        store = db.get_store(conn, order.store_id)
        as_of = db.world_asof(conn)
        store_product_ids = {p.id for p in db.list_products(conn, order.store_id)}

    facts = load_facts()
    override = store.return_window_days_override if store else None
    window_days = effective_return_window_days(facts["return_window_days"], override)
    window_source = "platform" if override is None else "store_override"
    policy_id = "cw-returns"
    if override is not None:
        store_doc = f"store-{store.slug}-policy"
        if any(doc.policy_id == store_doc for doc in load_policy_docs()):
            policy_id = store_doc

    delivered = order.delivered_at
    days_since = (as_of - delivered).days if delivered else None
    deadline = (delivered + timedelta(days=window_days)) if delivered else None

    issues = []
    if order.status == "delivered" and delivered is None:
        issues.append("the order is marked delivered but has no delivery date")
    if order.shipped_at and delivered and order.shipped_at > delivered:
        issues.append("the recorded ship date is after the delivery date")
    if delivered and delivered > as_of:
        issues.append("the delivery date is in the future")
    if order.product_id not in store_product_ids:
        issues.append("the order's product does not belong to the order's store")
    computed = is_refund_eligible(
        status=order.status, delivered_at=delivered, as_of=as_of, return_window_days=window_days
    )
    if computed != order.refund_eligible:
        issues.append("the stamped eligibility disagrees with the delivery date and window")
    data_issue = None
    if issues:
        data_issue = (
            "Inconsistent order record: " + "; ".join(issues)
            + ". Do not state a return deadline; escalate to a human."
        )

    if data_issue:
        reason = data_issue
    elif order.status != "delivered":
        reason = f"Order {order_id} is {order.status!r}; only delivered orders can be refunded ({policy_id})."
    elif order.refund_eligible:
        reason = (
            f"Delivered {delivered.isoformat()}, {_days_ago(days_since)}; within the "
            f"{window_days}-day window ({policy_id}), which closes on {deadline.isoformat()}."
        )
    else:
        reason = (
            f"Delivered {delivered.isoformat()}, {_days_ago(days_since)}; the "
            f"{window_days}-day window ({policy_id}) closed on {deadline.isoformat()}."
        )

    threshold = facts["refund_auto_approve_threshold_usd"]
    return {
        "ok": True,
        "order_id": order_id,
        "eligible": order.refund_eligible,
        "status": order.status,
        "delivered_at": delivered.isoformat() if delivered else None,
        "days_since_delivery": days_since,
        "window_days": window_days,
        "window_source": window_source,
        "deadline": deadline.isoformat() if deadline else None,
        "policy_id": policy_id,
        "total_usd": order.total_usd,
        "needs_approval": refund_needs_approval(order.total_usd, threshold),
        "reason": reason,
        "data_issue": data_issue,
    }


def _days_ago(days: int) -> str:
    return "1 day ago" if days == 1 else f"{days} days ago"
