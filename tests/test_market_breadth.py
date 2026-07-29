"""
Unit test cho core/market_breadth.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.market_breadth import (
    BREADTH_THRESHOLDS,
    InsufficientDataError,
    aggregate_layer3_indicators_for_group,
    calculate_adx,
    calculate_advance_decline_line,
    calculate_atr,
    calculate_bollinger_band_width,
    calculate_breadth_trend,
    calculate_ema200_breadth,
    calculate_ema200_deviation,
    calculate_new_high_low_ratio,
    calculate_volume_ratio,
    classify_breadth_label,
    detect_bearish_divergence,
    detect_bullish_divergence,
    detect_ma_cross,
)


def _snap(close, ema200):
    return {"close": close, "ema200": ema200}


def _make_df(closes, highs=None, lows=None, volumes=None, n=None):
    n = n or len(closes)
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n),
        "open": closes,
        "high": highs or [c * 1.01 for c in closes],
        "low": lows or [c * 0.99 for c in closes],
        "close": closes,
        "volume": volumes or [1000] * n,
    })


# ==============================================================================
# Test: calculate_ema200_breadth
# ==============================================================================

class TestCalculateEMA200Breadth:
    def test_known_ratio(self):
        snapshots = [
            _snap(120, 100), _snap(110, 100), _snap(90, 100), _snap(80, 100),
        ]
        result = calculate_ema200_breadth(snapshots)
        assert result["breadth_pct"] == pytest.approx(50.0)
        assert result["n_above"] == 2
        assert result["n_valid"] == 4

    def test_excludes_symbols_without_ema200(self):
        snapshots = [_snap(120, 100), {"close": 50, "ema200": None}]
        result = calculate_ema200_breadth(snapshots)
        assert result["n_valid"] == 1

    def test_empty_returns_none(self):
        result = calculate_ema200_breadth([])
        assert result["breadth_pct"] is None


# ==============================================================================
# Test: calculate_ema200_deviation
# ==============================================================================

class TestCalculateEMA200Deviation:
    def test_known_average_deviation(self):
        snapshots = [_snap(110, 100), _snap(90, 100)]  # +10%, -10%
        result = calculate_ema200_deviation(snapshots)
        assert result == pytest.approx(0.0)

    def test_all_positive_deviation(self):
        snapshots = [_snap(120, 100), _snap(110, 100)]  # +20%, +10%
        result = calculate_ema200_deviation(snapshots)
        assert result == pytest.approx(15.0)

    def test_empty_returns_none(self):
        assert calculate_ema200_deviation([]) is None


# ==============================================================================
# Test: classify_breadth_label
# ==============================================================================

class TestClassifyBreadthLabel:
    def test_extreme_uptrend(self):
        assert classify_breadth_label(85.0) == "uptrend_extreme"

    def test_uptrend(self):
        assert classify_breadth_label(70.0) == "uptrend"

    def test_sideway_mid_range(self):
        assert classify_breadth_label(50.0) == "sideway"

    def test_downtrend(self):
        assert classify_breadth_label(30.0) == "downtrend"

    def test_extreme_downtrend(self):
        assert classify_breadth_label(15.0) == "downtrend_extreme"

    def test_uptrend_downgraded_to_sideway_when_decreasing(self):
        assert classify_breadth_label(70.0, breadth_trend="decreasing") == "sideway"

    def test_downtrend_downgraded_to_sideway_when_increasing(self):
        assert classify_breadth_label(30.0, breadth_trend="increasing") == "sideway"

    def test_none_defaults_to_sideway(self):
        assert classify_breadth_label(None) == "sideway"

    def test_thresholds_constants_match_spec(self):
        assert BREADTH_THRESHOLDS["uptrend"] == 60.0
        assert BREADTH_THRESHOLDS["downtrend"] == 40.0
        assert BREADTH_THRESHOLDS["uptrend_extreme"] == 80.0
        assert BREADTH_THRESHOLDS["downtrend_extreme"] == 20.0


# ==============================================================================
# Test: calculate_breadth_trend
# ==============================================================================

class TestCalculateBreadthTrend:
    def test_increasing_trend(self):
        history = [40, 42, 45, 48, 52]
        assert calculate_breadth_trend(history) == "increasing"

    def test_decreasing_trend(self):
        history = [60, 55, 50, 45, 40]
        assert calculate_breadth_trend(history) == "decreasing"

    def test_flat_trend(self):
        history = [50, 51, 49, 50, 50]
        assert calculate_breadth_trend(history) == "flat"

    def test_insufficient_history_returns_none(self):
        assert calculate_breadth_trend([50]) is None


# ==============================================================================
# Test: detect_ma_cross
# ==============================================================================

class TestDetectMACross:
    def test_golden_cross_detected(self):
        fast = pd.Series([95, 105])   # cắt lên
        slow = pd.Series([100, 100])
        assert detect_ma_cross(fast, slow) == "golden_cross"

    def test_death_cross_detected(self):
        fast = pd.Series([105, 95])   # cắt xuống
        slow = pd.Series([100, 100])
        assert detect_ma_cross(fast, slow) == "death_cross"

    def test_no_cross(self):
        fast = pd.Series([105, 106])
        slow = pd.Series([100, 100])
        assert detect_ma_cross(fast, slow) == "none"


# ==============================================================================
# Test: calculate_adx
# ==============================================================================

class TestCalculateATR:
    def test_constant_high_low_range_yields_stable_atr(self):
        # High-Low luôn = 2, không có gap qua đêm -> ATR hội tụ về 2
        n = 30
        closes = [100.0] * n
        df = _make_df(closes, highs=[101.0] * n, lows=[99.0] * n)
        atr = calculate_atr(df, period=14)
        assert atr.iloc[-1] == pytest.approx(2.0, abs=0.01)

    def test_larger_true_range_yields_higher_atr(self):
        n = 30
        calm_df = _make_df([100.0] * n, highs=[100.5] * n, lows=[99.5] * n)
        volatile_df = _make_df([100.0] * n, highs=[105.0] * n, lows=[95.0] * n)
        calm_atr = calculate_atr(calm_df, period=14).iloc[-1]
        volatile_atr = calculate_atr(volatile_df, period=14).iloc[-1]
        assert volatile_atr > calm_atr

    def test_raises_for_insufficient_data(self):
        df = _make_df([100, 101, 102])
        with pytest.raises(InsufficientDataError):
            calculate_atr(df, period=14)


class TestCalculateADX:
    def test_strong_uptrend_yields_high_adx(self):
        # Xu hướng tăng đều, mạnh, rõ ràng -> ADX phải > 25 (ngưỡng xu hướng mạnh)
        n = 60
        closes = [100 + i * 2 for i in range(n)]
        df = _make_df(
            closes,
            highs=[c + 1 for c in closes],
            lows=[c - 1 for c in closes],
        )
        adx = calculate_adx(df, period=14)
        assert adx.iloc[-1] > 25

    def test_sideways_choppy_yields_low_adx(self):
        # Giá dao động lên xuống không xu hướng -> ADX phải thấp
        n = 60
        closes = [100 + (5 if i % 2 == 0 else -5) for i in range(n)]
        df = _make_df(
            closes,
            highs=[c + 1 for c in closes],
            lows=[c - 1 for c in closes],
        )
        adx = calculate_adx(df, period=14)
        assert adx.iloc[-1] < 25

    def test_raises_for_insufficient_data(self):
        df = _make_df([100, 101, 102])
        with pytest.raises(InsufficientDataError):
            calculate_adx(df, period=14)


# ==============================================================================
# Test: calculate_bollinger_band_width
# ==============================================================================

class TestCalculateBollingerBandWidth:
    def test_flat_prices_yield_near_zero_width(self):
        df = _make_df([100.0] * 25)
        width = calculate_bollinger_band_width(df, period=20)
        assert width.iloc[-1] == pytest.approx(0.0, abs=0.01)

    def test_volatile_prices_yield_wider_band(self):
        closes = [100 + (10 if i % 2 == 0 else -10) for i in range(25)]
        df = _make_df(closes)
        width = calculate_bollinger_band_width(df, period=20)
        assert width.iloc[-1] > 10

    def test_raises_for_insufficient_data(self):
        df = _make_df([100, 101, 102])
        with pytest.raises(InsufficientDataError):
            calculate_bollinger_band_width(df, period=20)


# ==============================================================================
# Test: calculate_volume_ratio
# ==============================================================================

class TestCalculateVolumeRatio:
    def test_known_ratio(self):
        volumes = [1000] * 20 + [3000]
        df = _make_df([100.0] * 21, volumes=volumes)
        ratio = calculate_volume_ratio(df, period=20)
        # volume_ma tại phiên cuối = trung bình 20 phiên GẦN NHẤT (bao gồm
        # chính phiên hiện tại) = (19*1000 + 3000)/20 = 1100
        # ratio = 3000 / 1100 = 2.727...
        expected_ma = (19 * 1000 + 3000) / 20
        assert ratio.iloc[-1] == pytest.approx(3000 / expected_ma, rel=0.01)

    def test_raises_for_insufficient_data(self):
        df = _make_df([100, 101])
        with pytest.raises(InsufficientDataError):
            calculate_volume_ratio(df, period=20)


# ==============================================================================
# Test: calculate_advance_decline_line
# ==============================================================================

class TestCalculateAdvanceDeclineLine:
    def test_known_cumulative_values(self):
        # 3 mã, 3 phiên: phiên 1: 2 tăng 1 giảm (+1); phiên 2: 1 tăng 2 giảm (-1); phiên 3: 3 tăng (+3)
        stock_a = pd.Series([1.0, -1.0, 1.0])
        stock_b = pd.Series([1.0, -1.0, 1.0])
        stock_c = pd.Series([-1.0, 1.0, 1.0])

        ad_line = calculate_advance_decline_line([stock_a, stock_b, stock_c])

        assert ad_line.iloc[0] == 1    # +1
        assert ad_line.iloc[1] == 0    # +1 + (-1) = 0
        assert ad_line.iloc[2] == 3    # 0 + 3 = 3

    def test_raises_for_empty_input(self):
        with pytest.raises(ValueError):
            calculate_advance_decline_line([])


# ==============================================================================
# Test: calculate_new_high_low_ratio
# ==============================================================================

class TestDetectBullishDivergence:
    def _make_divergence_df(self):
        """Dữ liệu ĐÃ DÒ KIỂM THỰC TẾ: đáy 1 (giá=159.2, RSI=13.6) rồi
        đáy 2 (giá=145.3 THẤP HƠN, nhưng RSI=25.1 CAO HƠN) -> phân kỳ
        tăng thật, không phải suy diễn lý thuyết.
        """
        preamble = [250, 252, 248, 251, 249, 253, 247, 250, 252, 248, 251, 249, 253, 250, 252]
        leg1 = [250 - i * 10 for i in range(10)]
        bounce = [160 + i * 8 for i in range(6)]
        pattern = [-6, -5, 4, -6, -5, 3, -6, -4, 3, -6, -5, 2, -6, -4, 2, -6, -5, 2, -6, -8]
        leg2 = []
        price = 208
        for d in pattern:
            price += d
            leg2.append(price)
        # Thêm vài phiên SAU đáy 2 để thuật toán xác nhận được đó thực sự
        # là đáy swing (cần đủ `swing_order` phiên phía sau để so sánh).
        after = [price + 5, price + 3, price + 6, price + 4]

        closes = preamble + leg1 + bounce + leg2 + after
        return pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": [c * 1.005 for c in closes],
            "low": [c * 0.995 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        })

    def test_detects_genuine_bullish_divergence(self):
        df = self._make_divergence_df()
        result = detect_bullish_divergence(df, rsi_period=14, lookback=90, swing_order=3)
        assert result["detected"] is True
        assert result["price_low_2"] < result["price_low_1"]
        assert result["rsi_low_2"] > result["rsi_low_1"]

    def test_no_divergence_for_pure_monotonic_downtrend(self):
        # Xu hướng giảm đều, không có đáy swing xen kẽ rõ ràng -> không phân kỳ
        n = 60
        closes = [200 - i * 2 for i in range(n)]
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": closes, "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes], "close": closes,
            "volume": [1000] * n,
        })
        result = detect_bullish_divergence(df, rsi_period=14, lookback=60, swing_order=3)
        assert result["detected"] is False

    def test_no_divergence_when_second_low_is_higher(self):
        # Đáy sau CAO HƠN đáy trước -> không phải downtrend tạo đáy mới, không tính là phân kỳ
        preamble = [100] * 15
        leg1 = [100 - i * 5 for i in range(10)]  # xuống 50
        bounce = [50 + i * 10 for i in range(10)]  # lên 150
        leg2_up = [150 - i * 2 for i in range(15)]  # xuống nhẹ, đáy vẫn cao hơn leg1
        closes = preamble + leg1 + bounce + leg2_up
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=len(closes)),
            "open": closes, "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes], "close": closes,
            "volume": [1000] * len(closes),
        })
        result = detect_bullish_divergence(df, rsi_period=14, lookback=60, swing_order=3)
        assert result["detected"] is False

    def test_insufficient_data_returns_not_detected_with_reason(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=10),
            "open": [100] * 10, "high": [101] * 10, "low": [99] * 10,
            "close": [100] * 10, "volume": [1000] * 10,
        })
        result = detect_bullish_divergence(df)
        assert result["detected"] is False
        assert "reason" in result


class TestCalculateNewHighLowRatio:
    def test_known_ratio(self):
        prices = {
            "A": pd.Series([10, 20, 30]),   # 30 là đỉnh mới
            "B": pd.Series([30, 20, 10]),   # 10 là đáy mới
            "C": pd.Series([10, 20, 15]),   # không phải đỉnh/đáy mới
        }
        result = calculate_new_high_low_ratio(prices, window=3)
        assert result["n_new_high"] == 1
        assert result["n_new_low"] == 1
        assert result["n_symbols"] == 3
        assert result["new_high_ratio"] == pytest.approx(33.33, abs=0.1)

    def test_raises_for_empty_input(self):
        with pytest.raises(ValueError):
            calculate_new_high_low_ratio({})


# ==============================================================================
# Test: aggregate_layer3_indicators_for_group
# ==============================================================================

def _make_trend_df(n=260, direction="up"):
    """Tạo OHLCV giả lập có xu hướng rõ ràng (tăng hoặc giảm đều) — đủ để
    tính MA50/200 cross, ADX, Band Width mà không lỗi thiếu dữ liệu.
    """
    if direction == "up":
        closes = [100 + i * 0.5 for i in range(n)]
    else:
        closes = [100 + (n - i) * 0.5 for i in range(n)]
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n),
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes,
        "volume": [1000] * n,
    })


class TestAggregateLayer3IndicatorsForGroup:
    def test_majority_uptrend_symbols_yield_golden_cross_consensus(self):
        # 3 mã có MA50 cắt lên MA200 rõ ràng (do giá tăng đều dài hạn),
        # dựng bằng cách nối 1 đoạn ngang rồi tăng mạnh cuối kỳ để tạo cắt lên
        def make_cross_up_df():
            flat = [100.0] * 210
            rising = [100 + i * 3 for i in range(2)]  # chỉ vừa đủ để cắt lên NGAY phiên cuối
            closes = flat + rising
            n = len(closes)
            return pd.DataFrame({
                "date": pd.bdate_range("2024-01-01", periods=n),
                "open": closes, "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes], "close": closes,
                "volume": [1000] * n,
            })

        ohlcv_by_symbol = {
            "AAA": make_cross_up_df(),
            "BBB": make_cross_up_df(),
            "CCC": _make_trend_df(direction="down"),  # thiểu số, không cùng chiều
        }
        result = aggregate_layer3_indicators_for_group(ohlcv_by_symbol)
        assert result["ma_cross"] == "golden_cross"

    def test_no_clear_majority_yields_none(self):
        ohlcv_by_symbol = {
            "AAA": _make_trend_df(direction="up"),
            "BBB": _make_trend_df(direction="down"),
        }
        result = aggregate_layer3_indicators_for_group(ohlcv_by_symbol)
        # Không có mã nào thực sự "cross" rõ ràng trong xu hướng đều (không
        # có đoạn phẳng trước đó để tạo điểm cắt) -> kỳ vọng "none"
        assert result["ma_cross"] == "none"

    def test_adx_and_band_width_are_averaged(self):
        ohlcv_by_symbol = {
            "AAA": _make_trend_df(direction="up"),
            "BBB": _make_trend_df(direction="up"),
        }
        result = aggregate_layer3_indicators_for_group(ohlcv_by_symbol)
        assert "adx" in result
        assert "band_width_percentile" in result
        assert result["adx"] > 0

    def test_skips_symbols_with_insufficient_data(self):
        short_df = _make_trend_df(n=10, direction="up")  # quá ít dữ liệu
        good_df = _make_trend_df(n=260, direction="up")
        result = aggregate_layer3_indicators_for_group({"AAA": short_df, "BBB": good_df})
        # Không lỗi, vẫn trả về kết quả dựa trên mã đủ dữ liệu (BBB)
        assert "adx" in result

    def test_empty_input_returns_none_ma_cross_without_error(self):
        result = aggregate_layer3_indicators_for_group({})
        assert result["ma_cross"] == "none"
        assert "adx" not in result


class TestDetectBearishDivergence:
    def _make_divergence_df(self):
        """Dữ liệu ĐÃ DÒ KIỂM THỰC TẾ: đỉnh 1 (giá=140.7, RSI=91.7) rồi
        đỉnh 2 (giá=154.8 CAO HƠN, nhưng RSI=75.9 THẤP HƠN) -> phân kỳ
        giảm thật.
        """
        preamble = [50, 52, 48, 51, 49, 53, 47, 50, 52, 48, 51, 49, 53, 50, 52]
        leg1 = [50 + i * 10 for i in range(10)]
        bounce_down = [140 - i * 8 for i in range(6)]
        pattern = [6, 5, -4, 6, 5, -3, 6, 4, -3, 6, 5, -2, 6, 4, -2, 6, 5, -2, 6, 8]
        leg2 = []
        price = 92
        for d in pattern:
            price += d
            leg2.append(price)
        after = [price - 5, price - 3, price - 6, price - 4]
        closes = preamble + leg1 + bounce_down + leg2 + after
        return pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": [c * 1.005 for c in closes],
            "low": [c * 0.995 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        })

    def test_detects_genuine_bearish_divergence(self):
        df = self._make_divergence_df()
        result = detect_bearish_divergence(df, rsi_period=14, lookback=90, swing_order=3)
        assert result["detected"] is True
        assert result["price_high_2"] > result["price_high_1"]
        assert result["rsi_high_2"] < result["rsi_high_1"]

    def test_no_divergence_for_pure_monotonic_uptrend(self):
        n = 60
        closes = [100 + i * 2 for i in range(n)]
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": closes, "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes], "close": closes,
            "volume": [1000] * n,
        })
        result = detect_bearish_divergence(df, rsi_period=14, lookback=60, swing_order=3)
        assert result["detected"] is False

    def test_insufficient_data_returns_not_detected_with_reason(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=10),
            "open": [100] * 10, "high": [101] * 10, "low": [99] * 10,
            "close": [100] * 10, "volume": [1000] * 10,
        })
        result = detect_bearish_divergence(df)
        assert result["detected"] is False
        assert "reason" in result
