"""Tests for the student-added check_refund_eligibility tool (HW1 Part A)."""

from __future__ import annotations

from agent import tools
from agent.agent import TOOLS_BY_ROLE, get_order_logic
from agent.auth import AuthContext

SHOPPER_1 = AuthContext(user_id=1, role="shopper")
SHOPPER_2 = AuthContext(user_id=2, role="shopper")
MERCHANT_STORE_1 = AuthContext(user_id=9001, role="merchant", store_id=1)
MERCHANT_STORE_2 = AuthContext(user_id=9002, role="merchant", store_id=2)
SUPPORT = AuthContext(user_id=9501, role="support")


def test_outside_platform_window_explains_and_cites(world: dict) -> None:
    result = tools.check_refund_eligibility(SHOPPER_1, 3980)
    assert result["ok"] is True
    assert result["eligible"] is False
    assert result["days_since_delivery"] == 45
    assert result["window_days"] == 30
    assert result["window_source"] == "platform"
    assert result["deadline"] == "2026-06-16"
    assert result["policy_id"] == "cw-returns"
    assert "cw-returns" in result["reason"]
    assert result["data_issue"] is None


def test_eligible_order_reports_approval_need(world: dict) -> None:
    small = tools.check_refund_eligibility(SHOPPER_1, 4127)
    assert small["eligible"] is True and small["needs_approval"] is False
    large = tools.check_refund_eligibility(SHOPPER_1, 4455)
    assert large["eligible"] is True and large["needs_approval"] is True


def test_store_override_window_and_policy(world: dict) -> None:
    # Order 2485 (shopper 1's Matte Vase) is from Juniper Home Goods, a 14-day store.
    result = tools.check_refund_eligibility(SHOPPER_1, 2485)
    assert result["window_days"] == 14
    assert result["window_source"] == "store_override"
    assert result["policy_id"] == "store-juniper-home-goods-policy"


def test_eligible_matches_stamped_flag_used_by_issue_refund(world: dict) -> None:
    for order_id in (3980, 4127, 4455, 2485):
        order = tools.check_refund_eligibility(SUPPORT, order_id)
        stamped = get_order_logic(SUPPORT, order_id)["order"]["refund_eligible"]
        assert order["eligible"] is stamped


def test_scope_is_enforced_before_revealing_anything(world: dict) -> None:
    for ctx in (SHOPPER_2, MERCHANT_STORE_2):
        denied = tools.check_refund_eligibility(ctx, 4127)
        assert denied == {
            "ok": False,
            "error": "permission_denied",
            "reason": denied["reason"],
        }
        assert "2026" not in denied["reason"] and "Blue Heron" not in denied["reason"]
    assert tools.check_refund_eligibility(MERCHANT_STORE_1, 4127)["ok"] is True
    assert tools.check_refund_eligibility(SUPPORT, 4127)["ok"] is True
    assert tools.check_refund_eligibility(SHOPPER_1, 999_999)["error"] == "not_found"


def test_inconsistent_records_flag_data_issue_without_deadline_claim(world: dict) -> None:
    for order_id in (8001, 8002, 8003):
        result = tools.check_refund_eligibility(SUPPORT, order_id)
        assert result["ok"] is True
        assert result["data_issue"], order_id
        assert "escalate" in result["data_issue"]
        assert result["reason"] == result["data_issue"]
    assert tools.check_refund_eligibility(SUPPORT, 8002)["deadline"] is None


def test_registered_for_every_role() -> None:
    for role, role_tools in TOOLS_BY_ROLE.items():
        assert "check_refund_eligibility" in {t.name for t in role_tools}, role
