"""
Unit test cho core/paper_portfolio.py

Viết test cho việc tính PnL và tỷ trọng với vài kịch bản giao dịch mẫu,
theo đúng yêu cầu dự án.
"""

from __future__ import annotations

import pytest

from core.paper_portfolio import (
    InsufficientCashError,
    InsufficientPositionError,
    InvalidTradeError,
    PaperPortfolio,
    create_portfolio,
)


# ==============================================================================
# Test: khởi tạo danh mục
# ==============================================================================

class TestCreatePortfolio:
    def test_initial_cash_set_correctly(self):
        portfolio = create_portfolio(1_000_000)
        assert portfolio.cash == 1_000_000
        assert portfolio.initial_cash == 1_000_000
        assert portfolio.positions == {}
        assert portfolio.trade_history == []

    def test_raises_for_non_positive_initial_cash(self):
        with pytest.raises(InvalidTradeError):
            create_portfolio(0)


# ==============================================================================
# Test: ghi nhận lệnh MUA
# ==============================================================================

class TestRecordTradeBuy:
    def test_single_buy_reduces_cash_and_creates_position(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=100, price=100, fee_pct=0.5)

        # cost=10,000; fee=50; total=10,050
        assert portfolio.cash == pytest.approx(1_000_000 - 10_050)
        assert "HPG" in portfolio.positions
        assert portfolio.positions["HPG"].qty == 100
        assert portfolio.positions["HPG"].avg_cost == pytest.approx(100.5)

    def test_multiple_buys_average_cost_correctly(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=100, price=100, fee_pct=0.5)
        portfolio.record_trade("HPG", "buy", qty=50, price=110, fee_pct=0.5)

        pos = portfolio.positions["HPG"]
        assert pos.qty == 150
        # avg_cost = (10,050 + 5,527.5) / 150 = 103.85
        assert pos.avg_cost == pytest.approx(103.85)

    def test_insufficient_cash_raises(self):
        portfolio = create_portfolio(1_000)
        with pytest.raises(InsufficientCashError):
            portfolio.record_trade("HPG", "buy", qty=100, price=100)

    def test_invalid_qty_raises(self):
        portfolio = create_portfolio(1_000_000)
        with pytest.raises(InvalidTradeError):
            portfolio.record_trade("HPG", "buy", qty=0, price=100)

    def test_invalid_price_raises(self):
        portfolio = create_portfolio(1_000_000)
        with pytest.raises(InvalidTradeError):
            portfolio.record_trade("HPG", "buy", qty=10, price=0)

    def test_invalid_side_raises(self):
        portfolio = create_portfolio(1_000_000)
        with pytest.raises(InvalidTradeError):
            portfolio.record_trade("HPG", "hold", qty=10, price=100)


# ==============================================================================
# Test: ghi nhận lệnh BÁN và tính realized PnL
# ==============================================================================

class TestRecordTradeSell:
    def test_partial_sell_computes_realized_pnl_correctly(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=100, price=100, fee_pct=0.5)
        portfolio.record_trade("HPG", "buy", qty=50, price=110, fee_pct=0.5)
        # avg_cost hiện tại = 103.85 (đã kiểm chứng ở test trên)

        record = portfolio.record_trade("HPG", "sell", qty=60, price=120, fee_pct=0.5)

        # proceeds=7,200; fee=36; net=7,164; realized_pnl=7,164-103.85*60=933.0
        assert record.realized_pnl == pytest.approx(933.0, abs=0.01)

    def test_partial_sell_reduces_qty_but_keeps_avg_cost(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=100, price=100, fee_pct=0.5)
        portfolio.record_trade("HPG", "buy", qty=50, price=110, fee_pct=0.5)
        portfolio.record_trade("HPG", "sell", qty=60, price=120, fee_pct=0.5)

        pos = portfolio.positions["HPG"]
        assert pos.qty == 90
        assert pos.avg_cost == pytest.approx(103.85)  # không đổi khi bán 1 phần

    def test_full_sell_removes_position(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=100, price=100)
        portfolio.record_trade("HPG", "sell", qty=100, price=110)

        assert "HPG" not in portfolio.positions

    def test_sell_more_than_held_raises(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=50, price=100)
        with pytest.raises(InsufficientPositionError):
            portfolio.record_trade("HPG", "sell", qty=100, price=110)

    def test_sell_symbol_not_held_raises(self):
        portfolio = create_portfolio(1_000_000)
        with pytest.raises(InsufficientPositionError):
            portfolio.record_trade("VNM", "sell", qty=10, price=100)


# ==============================================================================
# Test: entry_range_at_signal được lưu lại đầy đủ
# ==============================================================================

class TestEntryRangeTracking:
    def test_entry_range_stored_in_trade_record(self):
        portfolio = create_portfolio(1_000_000)
        record = portfolio.record_trade(
            "HPG", "buy", qty=10, price=100,
            entry_range_at_signal={"low": 95, "high": 102},
        )
        assert record.entry_range_at_signal == {"low": 95, "high": 102}

    def test_entry_range_retrievable_from_history(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade(
            "HPG", "buy", qty=10, price=105,
            entry_range_at_signal={"low": 95, "high": 102},
        )
        history = portfolio.get_trade_history("HPG")
        assert history[0].entry_range_at_signal == {"low": 95, "high": 102}
        # Có thể đối chiếu: giá vào lệnh (105) đã VƯỢT cận trên khuyến nghị (102)
        assert history[0].price > history[0].entry_range_at_signal["high"]


# ==============================================================================
# Test: get_portfolio_snapshot — NAV, PnL chưa thực hiện, tỷ trọng
# ==============================================================================

class TestGetPortfolioSnapshot:
    def test_snapshot_with_no_positions_equals_cash(self):
        portfolio = create_portfolio(1_000_000)
        snapshot = portfolio.get_portfolio_snapshot(current_prices={})
        assert snapshot["nav"] == pytest.approx(1_000_000)
        assert snapshot["total_stock_weight_pct"] == pytest.approx(0.0)
        assert snapshot["positions"] == []

    def test_snapshot_computes_unrealized_pnl_and_weight(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=100, price=100, fee_pct=0.5)
        portfolio.record_trade("HPG", "buy", qty=50, price=110, fee_pct=0.5)
        portfolio.record_trade("HPG", "sell", qty=60, price=120, fee_pct=0.5)
        # cash còn lại = 984,422.5 + 7,164 = 991,586.5 ; qty còn = 90, avg_cost=103.85

        snapshot = portfolio.get_portfolio_snapshot(current_prices={"HPG": 130})

        pos = snapshot["positions"][0]
        assert pos["qty"] == 90
        assert pos["market_value"] == pytest.approx(90 * 130)
        assert pos["unrealized_pnl"] == pytest.approx(90 * 130 - 103.85 * 90, abs=0.01)

        expected_nav = 991_586.5 + 90 * 130
        assert snapshot["nav"] == pytest.approx(expected_nav, abs=0.5)
        assert snapshot["total_realized_pnl"] == pytest.approx(933.0, abs=0.01)

    def test_weight_pct_sums_correctly_across_positions(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=100, price=50)   # 5,000
        portfolio.record_trade("VNM", "buy", qty=50, price=100)   # 5,000

        snapshot = portfolio.get_portfolio_snapshot(
            current_prices={"HPG": 50, "VNM": 100}
        )
        total_weight = sum(p["weight_pct"] for p in snapshot["positions"])
        assert total_weight == pytest.approx(snapshot["total_stock_weight_pct"], abs=0.01)

    def test_missing_price_falls_back_to_avg_cost_with_warning(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=100, price=100)

        snapshot = portfolio.get_portfolio_snapshot(current_prices={})  # thiếu giá HPG

        assert snapshot["positions"][0]["current_price"] == pytest.approx(100)
        assert len(snapshot["warnings"]) > 0

    def test_total_return_pct_reflects_gain(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=100, price=100)
        snapshot = portfolio.get_portfolio_snapshot(current_prices={"HPG": 150})
        # NAV tăng do giá tăng mạnh -> total_return_pct phải dương
        assert snapshot["total_return_pct"] > 0


# ==============================================================================
# Test: so sánh tỷ trọng thực tế với khuyến nghị (capital_allocator)
# ==============================================================================

class TestCompareToTargetAllocation:
    def test_flags_deviation_exceeding_threshold(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=100, price=100)  # 10,000 / ~1,000,000 = 1%

        result = portfolio.compare_to_target_allocation(
            current_prices={"HPG": 100}, target_pct=80.0, deviation_threshold_pct=10.0
        )
        assert result["exceeds_threshold"] is True

    def test_no_flag_when_within_threshold(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=100, price=100)

        result = portfolio.compare_to_target_allocation(
            current_prices={"HPG": 100}, target_pct=1.0, deviation_threshold_pct=10.0
        )
        assert result["exceeds_threshold"] is False


# ==============================================================================
# Test: lịch sử giao dịch
# ==============================================================================

class TestTradeHistory:
    def test_filters_by_symbol(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=10, price=100)
        portfolio.record_trade("VNM", "buy", qty=5, price=200)

        hpg_history = portfolio.get_trade_history("HPG")
        assert len(hpg_history) == 1
        assert hpg_history[0].symbol == "HPG"

    def test_returns_all_when_no_symbol_filter(self):
        portfolio = create_portfolio(1_000_000)
        portfolio.record_trade("HPG", "buy", qty=10, price=100)
        portfolio.record_trade("VNM", "buy", qty=5, price=200)

        all_history = portfolio.get_trade_history()
        assert len(all_history) == 2
