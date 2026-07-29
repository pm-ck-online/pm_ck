"""
Unit test cho core/pattern_detector.py

Gồm 2 bộ dữ liệu giả lập chính theo đúng yêu cầu:
- Một chuỗi giá có biên độ thu hẹp rõ rệt (PHẢI nhận diện được).
- Một chuỗi giá dao động không thu hẹp / ngẫu nhiên (KHÔNG được báo dương
  tính giả).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.pattern_detector import detect_narrowing_pattern


def _make_segmented_df(
    segment_amplitudes_pct: list[float],
    rows_per_segment: int = 60,
    start_date: str = "2024-01-01",
    base_low: float = 100.0,
) -> pd.DataFrame:
    """Tạo DataFrame OHLCV mà mỗi đoạn (segment) có biên độ % xác định
    trước — giúp kiểm chứng chính xác logic tính toán mà không phụ thuộc
    ngẫu nhiên.

    Với mỗi đoạn có % biên độ mong muốn `a`, đặt low cố định = base_low,
    high = base_low * (1 + a/100) cho MỌI phiên trong đoạn đó — nhờ vậy
    % biên độ tính ra đúng bằng giá trị `a` truyền vào, không lệch do làm
    tròn hay biến động ngẫu nhiên.
    """
    n_segments = len(segment_amplitudes_pct)
    total_rows = rows_per_segment * n_segments
    dates = pd.bdate_range(start=start_date, periods=total_rows)

    highs = []
    lows = []
    for amp in segment_amplitudes_pct:
        high_val = base_low * (1 + amp / 100.0)
        highs.extend([high_val] * rows_per_segment)
        lows.extend([base_low] * rows_per_segment)

    closes = [(h + l) / 2 for h, l in zip(highs, lows)]

    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000] * total_rows,
    })


def _make_random_walk_df(
    n_rows: int = 240,
    start_date: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Tạo chuỗi giá dao động ngẫu nhiên (random walk), KHÔNG có mẫu hình
    thu hẹp biên độ nào — dùng để kiểm tra không báo dương tính giả.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_rows)

    price = 100.0
    closes = []
    for _ in range(n_rows):
        price += rng.normal(0, 3.0)  # bước ngẫu nhiên biên độ lớn, không thu hẹp
        price = max(price, 10.0)
        closes.append(price)

    highs = [c * 1.03 for c in closes]
    lows = [c * 0.97 for c in closes]

    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000] * n_rows,
    })


# ==============================================================================
# Test: Nhận diện đúng khi biên độ thu hẹp rõ rệt (PHẢI phát hiện được)
# ==============================================================================

class TestDetectsNarrowingPattern:
    def test_detects_clear_narrowing_pattern(self):
        # Ví dụ minh họa trong yêu cầu dự án: 20% -> 10% -> 3-5%
        df = _make_segmented_df([20, 10, 5, 3], rows_per_segment=60)
        result = detect_narrowing_pattern(df, scan_months_range=(10, 30), n_segments=4)

        assert result is not None
        assert len(result["segments"]) == 4
        amplitudes = [s["amplitude_pct"] for s in result["segments"]]
        assert amplitudes == pytest.approx([20, 10, 5, 3], abs=0.01)

    def test_confidence_in_valid_range(self):
        df = _make_segmented_df([20, 10, 5, 3], rows_per_segment=60)
        result = detect_narrowing_pattern(df, scan_months_range=(10, 30), n_segments=4)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_accumulation_high_matches_last_segment(self):
        df = _make_segmented_df([20, 10, 5, 3], rows_per_segment=60)
        result = detect_narrowing_pattern(df, scan_months_range=(10, 30), n_segments=4)
        assert result["accumulation_high"] == pytest.approx(
            result["segments"][-1]["high"]
        )

    def test_symbol_passthrough(self):
        df = _make_segmented_df([20, 10, 5, 3], rows_per_segment=60)
        result = detect_narrowing_pattern(df, symbol="HPG")
        assert result["symbol"] == "HPG"

    def test_segments_are_chronologically_ordered(self):
        df = _make_segmented_df([20, 10, 5, 3], rows_per_segment=60)
        result = detect_narrowing_pattern(df, n_segments=4)
        end_dates = [s["end_date"] for s in result["segments"]]
        assert end_dates == sorted(end_dates)

    def test_longer_formation_time_yields_higher_confidence(self):
        # Cùng mức độ thu hẹp, nhưng khoảng thời gian quét được dài hơn
        # (gần mốc scan_months_max) phải cho confidence cao hơn.
        short_df = _make_segmented_df([20, 10, 5, 3], rows_per_segment=55)  # ~11 tháng
        long_df = _make_segmented_df([20, 10, 5, 3], rows_per_segment=150)  # ~30 tháng

        short_result = detect_narrowing_pattern(short_df, scan_months_range=(10, 30))
        long_result = detect_narrowing_pattern(long_df, scan_months_range=(10, 30))

        assert short_result is not None
        assert long_result is not None
        assert long_result["confidence"] > short_result["confidence"]


# ==============================================================================
# Test: KHÔNG báo dương tính giả với dữ liệu không có mẫu hình
# ==============================================================================

class TestDoesNotDetectFalsePositive:
    def test_random_walk_does_not_produce_narrowing_pattern(self):
        df = _make_random_walk_df(n_rows=240, seed=42)
        result = detect_narrowing_pattern(df, scan_months_range=(10, 30), n_segments=4)
        assert result is None

    def test_widening_amplitude_returns_none(self):
        # Biên độ MỞ RỘNG dần (ngược lại với thu hẹp) -> không được nhận diện
        df = _make_segmented_df([3, 5, 10, 20], rows_per_segment=60)
        result = detect_narrowing_pattern(df, scan_months_range=(10, 30), n_segments=4)
        assert result is None

    def test_non_monotonic_amplitude_returns_none(self):
        # Biên độ lên xuống thất thường, không có xu hướng thu hẹp rõ ràng
        df = _make_segmented_df([5, 15, 8, 20], rows_per_segment=60)
        result = detect_narrowing_pattern(df, scan_months_range=(10, 30), n_segments=4)
        assert result is None


# ==============================================================================
# Test: Dữ liệu lịch sử không đủ dài
# ==============================================================================

class TestInsufficientHistory:
    def test_returns_none_when_less_than_min_scan_months(self):
        # Chỉ có ~3 tháng dữ liệu, dưới ngưỡng scan_months_min=10
        df = _make_segmented_df([20, 10, 5, 3], rows_per_segment=15)  # ~3 tháng
        result = detect_narrowing_pattern(df, scan_months_range=(10, 30), n_segments=4)
        assert result is None


# ==============================================================================
# Test: Xác thực đầu vào
# ==============================================================================

class TestInputValidation:
    def test_missing_required_columns_raises(self):
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=10)})
        with pytest.raises(ValueError):
            detect_narrowing_pattern(df)

    def test_n_segments_less_than_2_raises(self):
        df = _make_segmented_df([20, 10], rows_per_segment=100)
        with pytest.raises(ValueError):
            detect_narrowing_pattern(df, n_segments=1)
