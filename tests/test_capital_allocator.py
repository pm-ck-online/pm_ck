"""
Unit test cho core/capital_allocator.py

Viết test riêng cho từng giai đoạn (uptrend/downtrend/sideway) và test
riêng trường hợp giá đã vượt ra ngoài entry_price_range, theo đúng yêu
cầu dự án.
"""

from __future__ import annotations

import pytest

from core.capital_allocator import (
    InvalidAllocationInputError,
    compute_max_position_size,
    compute_stop_loss,
    get_allocation_recommendation,
)


# ==============================================================================
# Test các hàm tính toán nhỏ
# ==============================================================================

class TestComputeStopLoss:
    def test_stop_loss_below_entry_low(self):
        stop_loss = compute_stop_loss(entry_low=100, stop_loss_pct=7.0)
        assert stop_loss == pytest.approx(93.0)

    def test_raises_for_non_positive_entry_low(self):
        with pytest.raises(InvalidAllocationInputError):
            compute_stop_loss(entry_low=0, stop_loss_pct=7.0)

    def test_raises_for_non_positive_stop_loss_pct(self):
        with pytest.raises(InvalidAllocationInputError):
            compute_stop_loss(entry_low=100, stop_loss_pct=0)


class TestComputeMaxPositionSize:
    def test_limited_by_risk(self):
        # NAV=1,000,000; risk 2% = 20,000; entry=110; stop_loss=93 -> risk/share=17
        # max_qty_by_risk = floor(20,000/17) = 1176
        qty = compute_max_position_size(
            nav=1_000_000, risk_per_trade_pct=2.0, entry_price=110, stop_loss=93,
        )
        assert qty == int(20_000 // 17)

    def test_limited_by_capital_budget(self):
        # Ngân sách vốn nhỏ hơn nhiều so với giới hạn rủi ro -> bị chặn bởi vốn
        qty = compute_max_position_size(
            nav=1_000_000, risk_per_trade_pct=2.0, entry_price=110, stop_loss=93,
            capital_budget=1_100,  # chỉ đủ mua 10 cổ phiếu
        )
        assert qty == 10

    def test_raises_when_entry_price_not_above_stop_loss(self):
        with pytest.raises(InvalidAllocationInputError):
            compute_max_position_size(
                nav=1_000_000, risk_per_trade_pct=2.0, entry_price=90, stop_loss=93,
            )


# ==============================================================================
# Test riêng giai đoạn UPTREND
# ==============================================================================

class TestUptrendRecommendation:
    def _base_context(self, **overrides):
        ctx = {"current_price": 105, "entry_low": 100, "entry_high": 110}
        ctx.update(overrides)
        return ctx

    def test_target_pct_within_uptrend_range(self):
        regime_result = {"regime": "uptrend", "confidence": 0.8, "affected_sectors": []}
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=self._base_context()
        )
        assert 70.0 <= result["target_pct"] <= 100.0

    def test_higher_confidence_yields_higher_target_pct(self):
        low_conf = get_allocation_recommendation(
            {"regime": "uptrend", "confidence": 0.1, "affected_sectors": []},
            nav=1_000_000_000, signal_price_context=self._base_context(),
        )
        high_conf = get_allocation_recommendation(
            {"regime": "uptrend", "confidence": 0.9, "affected_sectors": []},
            nav=1_000_000_000, signal_price_context=self._base_context(),
        )
        assert high_conf["target_pct"] > low_conf["target_pct"]

    def test_tranches_are_30_50_20(self):
        regime_result = {"regime": "uptrend", "confidence": 0.5, "affected_sectors": []}
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=self._base_context()
        )
        assert result["tranches"] == [30, 50, 20]

    def test_leading_sector_prioritized_when_not_cautioned(self):
        regime_result = {"regime": "uptrend", "confidence": 0.6, "affected_sectors": []}
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000,
            signal_price_context=self._base_context(sector="banking"),
        )
        assert any("dẫn dắt" in n for n in result["notes"])

    def test_leading_sector_not_prioritized_when_cautioned_by_macro(self):
        regime_result = {
            "regime": "uptrend", "confidence": 0.6, "affected_sectors": ["banking"],
        }
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000,
            signal_price_context=self._base_context(sector="banking"),
        )
        assert any("THẬN TRỌNG" in n for n in result["notes"])
        assert not any(
            "ưu tiên tìm cơ hội giải ngân" in n for n in result["notes"]
        )

    def test_max_position_size_positive_when_valid(self):
        regime_result = {"regime": "uptrend", "confidence": 0.7, "affected_sectors": []}
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=self._base_context()
        )
        assert result["max_position_size"] > 0

    def test_stop_loss_below_entry_low(self):
        regime_result = {"regime": "uptrend", "confidence": 0.7, "affected_sectors": []}
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=self._base_context()
        )
        assert result["stop_loss"] < result["entry_price_range"]["low"]


# ==============================================================================
# Test riêng giai đoạn DOWNTREND
# ==============================================================================

class TestDowntrendRecommendation:
    def _base_context(self, **overrides):
        ctx = {"current_price": 45, "entry_low": 40, "entry_high": 48}
        ctx.update(overrides)
        return ctx

    def test_no_bullish_divergence_blocks_recommendation(self):
        regime_result = {"regime": "downtrend", "confidence": 0.5, "affected_sectors": []}
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000,
            signal_price_context=self._base_context(has_bullish_divergence=False),
        )
        assert result["target_pct"] == 0.0
        assert result["max_position_size"] == 0
        assert any("bắt đáy" in n for n in result["notes"])

    def test_bullish_divergence_allows_cautious_recommendation(self):
        regime_result = {"regime": "downtrend", "confidence": 0.5, "affected_sectors": []}
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000,
            signal_price_context=self._base_context(has_bullish_divergence=True),
        )
        assert 10.0 <= result["target_pct"] <= 30.0
        assert result["max_position_size"] > 0

    def test_defensive_sector_prioritized(self):
        regime_result = {"regime": "downtrend", "confidence": 0.5, "affected_sectors": []}
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000,
            signal_price_context=self._base_context(
                has_bullish_divergence=True, sector="healthcare"
            ),
        )
        assert any("phòng thủ" in n for n in result["notes"])

    def test_target_pct_capped_low_range(self):
        regime_result = {"regime": "downtrend", "confidence": 0.9, "affected_sectors": []}
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000,
            signal_price_context=self._base_context(has_bullish_divergence=True),
        )
        assert result["target_pct"] <= 30.0


# ==============================================================================
# Test riêng giai đoạn SIDEWAY
# ==============================================================================

class TestSidewayRecommendation:
    def _base_context(self, **overrides):
        ctx = {"current_price": 52, "entry_low": 50, "entry_high": 55}
        ctx.update(overrides)
        return ctx

    def test_target_pct_within_sideway_range(self):
        regime_result = {"regime": "sideway", "confidence": 0.5, "affected_sectors": []}
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=self._base_context()
        )
        assert 30.0 <= result["target_pct"] <= 50.0

    def test_notes_mention_value_or_cash(self):
        regime_result = {"regime": "sideway", "confidence": 0.5, "affected_sectors": []}
        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=self._base_context()
        )
        assert any("giá trị" in n or "tiền mặt" in n for n in result["notes"])


# ==============================================================================
# Test riêng: giá đã vượt ra ngoài entry_price_range
# ==============================================================================

class TestEntryPriceRangeExceeded:
    def test_current_price_above_entry_high_blocks_uptrend_recommendation(self):
        regime_result = {"regime": "uptrend", "confidence": 0.7, "affected_sectors": []}
        ctx = {"current_price": 150, "entry_low": 100, "entry_high": 110}  # giá đã chạy xa

        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=ctx
        )

        assert result["target_pct"] == 0.0
        assert result["max_position_size"] == 0
        assert any("KHÔNG CÒN VÙNG ENTRY HỢP LỆ" in n for n in result["notes"])

    def test_entry_price_range_still_returned_even_when_exceeded(self):
        # Dù không còn hợp lệ để mua, vẫn phải trả về entry_price_range để
        # người dùng tham khảo (không được bỏ trống trường này).
        regime_result = {"regime": "uptrend", "confidence": 0.7, "affected_sectors": []}
        ctx = {"current_price": 150, "entry_low": 100, "entry_high": 110}

        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=ctx
        )

        assert result["entry_price_range"] == {"low": 100, "high": 110}
        assert result["stop_loss"] is not None

    def test_current_price_below_entry_low_still_allows_recommendation(self):
        # Giá CHƯA tới vùng entry (thấp hơn) -> vẫn là cơ hội hợp lệ trong
        # tương lai, chỉ cần ghi chú, KHÔNG chặn khuyến nghị.
        regime_result = {"regime": "uptrend", "confidence": 0.7, "affected_sectors": []}
        ctx = {"current_price": 90, "entry_low": 100, "entry_high": 110}

        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=ctx
        )

        assert result["target_pct"] > 0.0
        assert any("THẤP HƠN" in n for n in result["notes"])


# ==============================================================================
# Test: giai đoạn thị trường chưa được xác nhận (regime=None)
# ==============================================================================

class TestUnconfirmedRegime:
    def test_returns_no_recommendation_when_regime_is_none(self):
        regime_result = {"regime": None, "confidence": 0.0, "affected_sectors": []}
        ctx = {"current_price": 100, "entry_low": 95, "entry_high": 105}

        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=ctx
        )

        assert result["target_pct"] == 0.0
        assert result["max_position_size"] == 0
        assert result["entry_price_range"] == {"low": 95, "high": 105}


# ==============================================================================
# Test: cảnh báo rủi ro toàn danh mục
# ==============================================================================

class TestPortfolioRiskWarning:
    def test_warns_when_total_risk_would_exceed_threshold(self):
        regime_result = {"regime": "uptrend", "confidence": 0.8, "affected_sectors": []}
        ctx = {"current_price": 105, "entry_low": 100, "entry_high": 110}

        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=ctx,
            existing_portfolio_risk_pct=19.0,  # + 2% risk_per_trade = 21% > 20% ngưỡng
        )

        assert any("VƯỢT" in n for n in result["notes"])

    def test_no_warning_when_within_threshold(self):
        regime_result = {"regime": "uptrend", "confidence": 0.8, "affected_sectors": []}
        ctx = {"current_price": 105, "entry_low": 100, "entry_high": 110}

        result = get_allocation_recommendation(
            regime_result, nav=1_000_000_000, signal_price_context=ctx,
            existing_portfolio_risk_pct=5.0,
        )

        assert not any("VƯỢT" in n for n in result["notes"])


# ==============================================================================
# Test: xác thực đầu vào
# ==============================================================================

class TestInputValidation:
    def test_raises_for_non_positive_nav(self):
        regime_result = {"regime": "uptrend", "confidence": 0.5, "affected_sectors": []}
        ctx = {"current_price": 100, "entry_low": 95, "entry_high": 105}
        with pytest.raises(InvalidAllocationInputError):
            get_allocation_recommendation(regime_result, nav=0, signal_price_context=ctx)

    def test_raises_for_missing_price_context_keys(self):
        regime_result = {"regime": "uptrend", "confidence": 0.5, "affected_sectors": []}
        ctx = {"current_price": 100}  # thiếu entry_low, entry_high
        with pytest.raises(InvalidAllocationInputError):
            get_allocation_recommendation(regime_result, nav=1_000_000, signal_price_context=ctx)

    def test_raises_when_entry_low_greater_than_entry_high(self):
        regime_result = {"regime": "uptrend", "confidence": 0.5, "affected_sectors": []}
        ctx = {"current_price": 100, "entry_low": 110, "entry_high": 100}
        with pytest.raises(InvalidAllocationInputError):
            get_allocation_recommendation(regime_result, nav=1_000_000, signal_price_context=ctx)

    def test_raises_for_invalid_regime_value(self):
        regime_result = {"regime": "sideways_typo", "confidence": 0.5, "affected_sectors": []}
        ctx = {"current_price": 100, "entry_low": 95, "entry_high": 105}
        with pytest.raises(InvalidAllocationInputError):
            get_allocation_recommendation(regime_result, nav=1_000_000, signal_price_context=ctx)
