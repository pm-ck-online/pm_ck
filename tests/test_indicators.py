"""
Unit test cho core/indicators.py

Dùng dữ liệu giá giả lập ĐÃ BIẾT TRƯỚC kết quả tính toán để kiểm chứng
công thức MA/EMA/volume MA, không phụ thuộc vào bất kỳ nguồn dữ liệu thật.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.indicators import (
    InsufficientDataError,
    calculate_ema,
    calculate_ma,
    calculate_rsi,
    calculate_volume_ma,
    get_indicator_snapshot,
    is_volume_breakout,
    resample_ohlcv,
)


def _make_df(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    volumes = volumes or [1000] * n
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n),
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": volumes,
    })


# ==============================================================================
# Test: calculate_ma (Simple Moving Average)
# ==============================================================================

class TestCalculateMA:
    def test_known_values_period_3(self):
        df = _make_df([1, 2, 3, 4, 5])
        result = calculate_ma(df, period=3)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx((1 + 2 + 3) / 3)
        assert result.iloc[3] == pytest.approx((2 + 3 + 4) / 3)
        assert result.iloc[4] == pytest.approx((3 + 4 + 5) / 3)

    def test_missing_column_raises(self):
        df = _make_df([1, 2, 3])
        with pytest.raises(ValueError):
            calculate_ma(df, period=2, column="khong_ton_tai")

    def test_missing_required_columns_raises(self):
        df = pd.DataFrame({"close": [1, 2, 3]})  # thiếu date, open, high, low, volume
        with pytest.raises(ValueError):
            calculate_ma(df, period=2)


# ==============================================================================
# Test: calculate_ema (Exponential Moving Average)
# ==============================================================================

class TestCalculateEMA:
    def test_known_values_period_2(self):
        # alpha = 2 / (span + 1) = 2/3 với span=2, adjust=False
        prices = [1, 2, 3, 4, 5]
        df = _make_df(prices)
        result = calculate_ema(df, period=2)

        alpha = 2 / 3
        expected = [None] * len(prices)
        ema_prev = prices[0]
        expected_values = [ema_prev]
        for p in prices[1:]:
            ema_prev = alpha * p + (1 - alpha) * ema_prev
            expected_values.append(ema_prev)

        # min_periods=2 -> phiên đầu tiên (index 0) phải là NaN
        assert pd.isna(result.iloc[0])
        for i in range(1, len(prices)):
            assert result.iloc[i] == pytest.approx(expected_values[i], rel=1e-6)

    def test_ema_reacts_faster_than_ma_to_recent_change(self):
        # Chuỗi giá tăng đột biến ở cuối -> EMA phải phản ứng nhanh hơn MA
        prices = [10] * 20 + [50]
        df = _make_df(prices)
        ma = calculate_ma(df, period=10).iloc[-1]
        ema = calculate_ema(df, period=10).iloc[-1]
        assert ema > ma


# ==============================================================================
# Test: calculate_volume_ma
# ==============================================================================

class TestCalculateVolumeMA:
    def test_known_values(self):
        volumes = [100, 200, 300, 400, 500]
        df = _make_df(closes=[1, 1, 1, 1, 1], volumes=volumes)
        result = calculate_volume_ma(df, period=3)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx((100 + 200 + 300) / 3)
        assert result.iloc[4] == pytest.approx((300 + 400 + 500) / 3)


# ==============================================================================
# Test: is_volume_breakout
# ==============================================================================

class TestIsVolumeBreakout:
    def test_detects_breakout_when_volume_spikes(self):
        # 20 phiên volume ổn định quanh 1000, phiên cuối đột biến 2000
        volumes = [1000] * 20 + [2500]
        df = _make_df(closes=[10] * 21, volumes=volumes)
        assert is_volume_breakout(df, multiplier=1.5, volume_ma_period=20) is True

    def test_no_breakout_for_normal_volume(self):
        volumes = [1000] * 20 + [1050]
        df = _make_df(closes=[10] * 21, volumes=volumes)
        assert is_volume_breakout(df, multiplier=1.5, volume_ma_period=20) is False

    def test_raises_when_insufficient_data(self):
        df = _make_df(closes=[10] * 5, volumes=[1000] * 5)
        with pytest.raises(InsufficientDataError):
            is_volume_breakout(df, volume_ma_period=20)


# ==============================================================================
# Test: get_indicator_snapshot
# ==============================================================================

class TestGetIndicatorSnapshot:
    def test_snapshot_keys_present(self):
        df = _make_df(closes=list(range(1, 261)), volumes=[1000] * 260)
        snapshot = get_indicator_snapshot(df)

        expected_keys = {
            "date", "close", "volume", "ma20", "ema50", "ema100", "ema200",
            "volume_ma_15", "volume_ma_20", "price_above_ema200",
            "is_volume_breakout",
        }
        assert expected_keys.issubset(snapshot.keys())

    def test_all_indicators_computed_with_enough_data(self):
        df = _make_df(closes=list(range(1, 261)), volumes=[1000] * 260)
        snapshot = get_indicator_snapshot(df)

        assert snapshot["ma20"] is not None
        assert snapshot["ema50"] is not None
        assert snapshot["ema100"] is not None
        assert snapshot["ema200"] is not None

    def test_ema200_none_when_insufficient_data(self):
        # Chỉ có 50 phiên -> chưa đủ dữ liệu tính EMA200
        df = _make_df(closes=list(range(1, 51)), volumes=[1000] * 50)
        snapshot = get_indicator_snapshot(df)
        assert snapshot["ema200"] is None

    def test_price_above_ema200_true_for_uptrend(self):
        # Giá tăng dần đều -> giá hiện tại chắc chắn cao hơn EMA200
        df = _make_df(closes=[float(i) for i in range(1, 261)], volumes=[1000] * 260)
        snapshot = get_indicator_snapshot(df)
        assert snapshot["price_above_ema200"] is True

    def test_price_above_ema200_false_for_downtrend(self):
        # Giá giảm dần đều -> giá hiện tại chắc chắn thấp hơn EMA200
        df = _make_df(closes=[float(i) for i in range(260, 0, -1)], volumes=[1000] * 260)
        snapshot = get_indicator_snapshot(df)
        assert snapshot["price_above_ema200"] is False

    def test_custom_config_periods_applied(self):
        df = _make_df(closes=list(range(1, 61)), volumes=[1000] * 60)
        custom_config = {
            "ma_short_period": 10,
            "ema_mid_periods": [20],
            "ema_long_period": 50,
            "volume_ma_periods": [10],
            "breakout_volume_multiplier": 1.5,
        }
        snapshot = get_indicator_snapshot(df, config=custom_config)

        assert "ema20" in snapshot
        assert "volume_ma_10" in snapshot
        # key "ema200" luôn đại diện cho "EMA dài hạn cấu hình được" — với
        # ema_long_period=50 và đủ 60 phiên dữ liệu, giá trị này PHẢI được
        # tính ra (không phải None), và phải khớp với calculate_ema(period=50)
        assert snapshot["ema200"] is not None
        expected_ema50 = calculate_ema(df, period=50).iloc[-1]
        assert snapshot["ema200"] == pytest.approx(expected_ema50)


# ==============================================================================
# Test: calculate_rsi (chỉ báo bổ sung)
# ==============================================================================

class TestCalculateRSI:
    def test_rsi_approaches_100_for_continuous_gains(self):
        # Giá tăng liên tục, không giảm phiên nào -> RSI phải tiệm cận 100
        df = _make_df(closes=list(range(1, 61)))
        rsi = calculate_rsi(df, period=14)
        assert rsi.iloc[-1] == pytest.approx(100.0, abs=0.01)

    def test_rsi_approaches_0_for_continuous_losses(self):
        # Giá giảm liên tục -> RSI phải tiệm cận 0
        df = _make_df(closes=list(range(60, 0, -1)))
        rsi = calculate_rsi(df, period=14)
        assert rsi.iloc[-1] == pytest.approx(0.0, abs=0.01)

    def test_rsi_stays_within_valid_range(self):
        # Giá dao động lên xuống thất thường -> RSI luôn phải nằm trong [0, 100]
        closes = [10 + (i % 7) - 3 + i * 0.1 for i in range(100)]
        df = _make_df(closes=closes)
        rsi = calculate_rsi(df, period=14).dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_first_period_rows_are_nan(self):
        df = _make_df(closes=list(range(1, 61)))
        rsi = calculate_rsi(df, period=14)
        assert rsi.iloc[:14].isna().all()
        assert not pd.isna(rsi.iloc[14])

    def test_raises_when_insufficient_data(self):
        df = _make_df(closes=list(range(1, 10)))  # chỉ 9 phiên, cần > 14
        with pytest.raises(InsufficientDataError):
            calculate_rsi(df, period=14)

    def test_raises_for_missing_required_columns(self):
        df = pd.DataFrame({"close": [1, 2, 3] * 10})
        with pytest.raises(ValueError):
            calculate_rsi(df, period=14)


# ==============================================================================
# Test: resample_ohlcv
# ==============================================================================

class TestResampleOHLCV:
    def test_day_timeframe_returns_unchanged(self):
        df = _make_df(closes=[10, 11, 12, 13, 14])
        result = resample_ohlcv(df, timeframe="day")
        assert len(result) == 5
        pd.testing.assert_series_equal(
            result["close"].reset_index(drop=True),
            df["close"].reset_index(drop=True),
        )

    def test_week_aggregation_known_values(self):
        # 2 tuần liên tiếp, mỗi tuần đủ 5 phiên (Thứ 2 -> Thứ 6)
        dates = pd.bdate_range("2026-01-05", periods=10)  # 2026-01-05 là Thứ Hai
        df = pd.DataFrame({
            "date": dates,
            "open": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            "high": [15, 15, 15, 15, 20, 25, 25, 25, 25, 30],
            "low": [9, 9, 9, 9, 9, 19, 19, 19, 19, 19],
            "close": [11, 12, 13, 14, 20, 21, 22, 23, 24, 29],
            "volume": [100, 100, 100, 100, 100, 200, 200, 200, 200, 200],
        })
        result = resample_ohlcv(df, timeframe="week")

        assert len(result) == 2
        # Tuần 1: open=phiên đầu(10), close=phiên cuối(20), high=max(20), low=min(9), volume=sum(500)
        assert result.iloc[0]["open"] == 10
        assert result.iloc[0]["close"] == 20
        assert result.iloc[0]["high"] == 20
        assert result.iloc[0]["low"] == 9
        assert result.iloc[0]["volume"] == 500

        # Tuần 2: open=20... wait -> phiên đầu tuần 2 là index 5 (open=20)
        assert result.iloc[1]["open"] == 20
        assert result.iloc[1]["close"] == 29
        assert result.iloc[1]["high"] == 30
        assert result.iloc[1]["low"] == 19
        assert result.iloc[1]["volume"] == 1000

    def test_month_aggregation_reduces_row_count(self):
        dates = pd.bdate_range("2026-01-01", periods=60)  # ~3 tháng dữ liệu
        df = _make_df(closes=list(range(60)))
        df["date"] = dates
        result = resample_ohlcv(df, timeframe="month")
        assert len(result) < len(df)
        assert len(result) <= 4  # tối đa khoảng 3-4 tháng

    def test_invalid_timeframe_raises(self):
        df = _make_df(closes=[10, 11, 12])
        with pytest.raises(ValueError):
            resample_ohlcv(df, timeframe="year")

    def test_raises_for_missing_required_columns(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError):
            resample_ohlcv(df, timeframe="week")
