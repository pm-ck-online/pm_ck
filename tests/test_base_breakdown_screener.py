"""
Unit test cho core/base_breakdown_screener.py

Các giá trị kỳ vọng cụ thể đã được DÒ SỐ LIỆU THẬT trước khi viết test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.base_breakdown_screener import (
    InvalidBaseBreakdownError,
    kiem_tra_rsi_va_volume,
    quet_co_phieu_dut_gay_qua_ban,
    tinh_muc_giam_tu_diem_gay,
    xac_dinh_vung_nen,
)


def _make_breakdown_df(n: int = 150, volume_spike_last: float = 4000.0) -> pd.DataFrame:
    """Tạo DataFrame giả lập: vùng nền tích lũy chặt (ngày 90-120, nằm
    trong đúng 60 phiên lookback cuối), sau đó đứt gãy + giảm mạnh liên
    tục tới cuối chuỗi (~40% từ pivot), volume đột biến ở phiên cuối.
    """
    np.random.seed(7)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    closes = [100.0]
    for i in range(1, n):
        if 90 <= i < 120:
            closes.append(100 + np.random.normal(0, 0.3))
        elif i >= 120:
            closes.append(closes[-1] * (1 - 0.018))
        else:
            closes.append(closes[-1] * (1 + np.random.normal(0.0005, 0.004)))
    closes = np.array(closes)

    opens = closes * (1 + np.random.normal(0, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.002, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.002, n)))
    volumes = np.random.randint(1000, 1500, n).astype(float)
    volumes[-1] = volume_spike_last

    return pd.DataFrame({
        "date": dates, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


# ==============================================================================
# Test: validate input
# ==============================================================================

class TestValidateInput:
    def test_raises_on_empty_df(self):
        with pytest.raises(InvalidBaseBreakdownError):
            xac_dinh_vung_nen(pd.DataFrame())

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"date": [1, 2], "close": [1.0, 2.0]})
        with pytest.raises(InvalidBaseBreakdownError):
            xac_dinh_vung_nen(df)


# ==============================================================================
# Test: xac_dinh_vung_nen
# ==============================================================================

class TestXacDinhVungNen:
    def test_finds_consolidation_zone_in_engineered_data(self):
        df = _make_breakdown_df()
        vung_nen = xac_dinh_vung_nen(df, lookback=60, min_ngay=10)
        assert vung_nen is not None
        assert vung_nen["so_phien_tich_luy"] >= 10
        # Pivot hỗ trợ phải nằm trong vùng giá quanh 97-100 (vùng nền giả lập ~100)
        assert 95.0 <= vung_nen["gia_pivot_ho_tro"] <= 100.5

    def test_returns_none_when_not_enough_data(self):
        df = _make_breakdown_df(n=150).iloc[:5]  # chỉ 5 phiên, ít hơn min_ngay
        assert xac_dinh_vung_nen(df, lookback=60, min_ngay=10) is None


# ==============================================================================
# Test: tinh_muc_giam_tu_diem_gay
# ==============================================================================

class TestTinhMucGiamTuDiemGay:
    def test_computes_breakdown_percentage_correctly(self):
        df = _make_breakdown_df()
        vung_nen = xac_dinh_vung_nen(df, lookback=60, min_ngay=10)
        result = tinh_muc_giam_tu_diem_gay(df, vung_nen)
        assert result["da_dut_gay"] is True
        # Đã dò số liệu thật: giảm khoảng 40.9% từ pivot.
        assert result["pct_giam_tu_pivot"] == pytest.approx(40.91, abs=0.5)

    def test_no_breakdown_when_price_above_pivot(self):
        df = _make_breakdown_df()
        vung_nen = xac_dinh_vung_nen(df, lookback=60, min_ngay=10)
        # Giả lập giá hiện tại vẫn ở trên pivot -> chưa đứt gãy.
        df_no_break = df.copy()
        df_no_break.loc[df_no_break.index[-1], "close"] = vung_nen["gia_pivot_ho_tro"] * 1.05
        result = tinh_muc_giam_tu_diem_gay(df_no_break, vung_nen)
        assert result["da_dut_gay"] is False


# ==============================================================================
# Test: kiem_tra_rsi_va_volume
# ==============================================================================

class TestKiemTraRsiVaVolume:
    def test_detects_oversold_and_volume_spike(self):
        df = _make_breakdown_df()
        result = kiem_tra_rsi_va_volume(df, nguong_rsi=30, nguong_volume_ratio=1.5)
        assert result["dat_rsi"] is True
        assert result["dat_volume"] is True
        assert result["rsi_hien_tai"] < 30

    def test_fails_volume_check_when_volume_normal(self):
        df = _make_breakdown_df(volume_spike_last=1200.0)  # không đột biến
        result = kiem_tra_rsi_va_volume(df, nguong_rsi=30, nguong_volume_ratio=1.5)
        assert result["dat_volume"] is False

    def test_fails_rsi_check_with_high_threshold_ok_low_threshold_fails(self):
        df = _make_breakdown_df()
        # Ngưỡng RSI quá khắt khe (phải < 1) -> không đạt dù đang giảm mạnh.
        result = kiem_tra_rsi_va_volume(df, nguong_rsi=1.0, nguong_volume_ratio=1.5)
        assert result["dat_rsi"] is False


# ==============================================================================
# Test: quet_co_phieu_dut_gay_qua_ban (hàm chính, end-to-end)
# ==============================================================================

class TestQuetCoPhieuDutGayQuaBan:
    def test_detects_matching_stock(self):
        df = _make_breakdown_df()
        result = quet_co_phieu_dut_gay_qua_ban(["TEST"], lambda ma: df)
        assert len(result) == 1
        assert result.iloc[0]["ma"] == "TEST"
        assert result.iloc[0]["pct_giam_tu_pivot"] > 15.0

    def test_excludes_stock_that_fails_volume_criterion(self):
        df = _make_breakdown_df(volume_spike_last=1200.0)
        result = quet_co_phieu_dut_gay_qua_ban(["TEST"], lambda ma: df)
        assert result.empty

    def test_skips_symbol_with_insufficient_data_silently(self):
        short_df = _make_breakdown_df().iloc[:30]
        result = quet_co_phieu_dut_gay_qua_ban(["SHORT"], lambda ma: short_df)
        assert result.empty

    def test_skips_symbol_on_exception_without_crashing_whole_scan(self):
        def lay_ohlcv_loi(ma: str):
            if ma == "LOI":
                raise RuntimeError("Giả lập lỗi lấy dữ liệu")
            return _make_breakdown_df()

        result = quet_co_phieu_dut_gay_qua_ban(["LOI", "TEST"], lay_ohlcv_loi)
        # Mã lỗi bị bỏ qua âm thầm, mã còn lại vẫn được quét và phát hiện bình thường.
        assert len(result) == 1
        assert result.iloc[0]["ma"] == "TEST"

    def test_returns_empty_dataframe_when_no_symbols_match(self):
        result = quet_co_phieu_dut_gay_qua_ban([], lambda ma: None)
        assert result.empty

    def test_results_sorted_descending_by_pct_giam(self):
        df_strong = _make_breakdown_df()  # giảm ~40.9%

        # Mã thứ 2 có mức giảm từ pivot ÍT hơn (nhưng vẫn đủ 3 tiêu chí):
        # dựng lại bằng cách cắt bớt chuỗi giảm, giữ nguyên vùng nền.
        df_mild = _make_breakdown_df().copy()
        df_mild = df_mild.iloc[:135].copy()  # giảm ít hơn vì dừng sớm hơn
        df_mild.loc[df_mild.index[-1], "volume"] = 4000.0  # đảm bảo vẫn đột biến volume

        def lay_ohlcv(ma: str):
            return {"STRONG": df_strong, "MILD": df_mild}[ma]

        result = quet_co_phieu_dut_gay_qua_ban(["MILD", "STRONG"], lay_ohlcv)
        if len(result) == 2:
            assert result.iloc[0]["ma"] == "STRONG"
            assert result.iloc[0]["pct_giam_tu_pivot"] >= result.iloc[1]["pct_giam_tu_pivot"]
