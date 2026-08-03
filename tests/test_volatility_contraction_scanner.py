"""
Unit test cho core/volatility_contraction_scanner.py

Các giá trị kỳ vọng cụ thể đã được DÒ SỐ LIỆU THẬT trước khi viết test
(chạy trực tiếp hàm với dữ liệu giả lập, xác nhận kết quả rồi mới cố định
làm assertion).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.volatility_contraction_scanner import (
    InvalidVolatilityContractionError,
    doi_chieu_voi_ma20,
    gan_nhan_bac_bien_do,
    rao_soat_mo_hinh_co_hep,
    tim_dinh_day_cuc_bo,
    tinh_bien_do_tung_chu_ky,
    tu_chon_khung_thoi_gian,
    xac_nhan_chuoi_co_hep,
)


def _make_vcp_df(n: int = 121) -> pd.DataFrame:
    """Tạo DataFrame giả lập có mô hình co hẹp biên độ RÕ RÀNG: 6 chu kỳ
    đỉnh-đáy liên tiếp với biên độ giảm dần đều: 22,16% -> 15,62% -> 12,4%
    -> 10,35% -> 8,23% -> 5,21% (đã dò số liệu thật, khớp đúng ví dụ
    trong tài liệu gốc >20% -> ~15% -> ~10% -> ~5%).
    """
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    key_points = [
        (0, 100.0), (20, 78.0), (40, 90.0), (60, 79.0), (80, 87.0), (100, 80.0), (120, 84.0),
    ]
    closes = np.interp(np.arange(n), [p[0] for p in key_points], [p[1] for p in key_points])
    highs = closes * 1.001
    lows = closes * 0.999
    volumes = np.full(n, 1000.0)
    return pd.DataFrame({
        "date": dates, "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


class TestValidateInput:
    def test_raises_on_empty_df(self):
        with pytest.raises(InvalidVolatilityContractionError):
            tim_dinh_day_cuc_bo(pd.DataFrame())

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"date": [1, 2], "close": [1.0, 2.0]})
        with pytest.raises(InvalidVolatilityContractionError):
            tim_dinh_day_cuc_bo(df)


class TestTimDinhDayCucBo:
    def test_detects_engineered_peaks_and_troughs(self):
        df = _make_vcp_df()
        diem = tim_dinh_day_cuc_bo(df, khoang_cach_toi_thieu=3)
        loai_theo_thu_tu = [d["loai"] for d in diem]
        # Phải xen kẽ đỉnh-đáy, không có 2 loại giống nhau liên tiếp.
        for a, b in zip(loai_theo_thu_tu, loai_theo_thu_tu[1:]):
            assert a != b
        assert len(diem) >= 6  # đủ 7 điểm cực trị đã thiết kế (0,20,40,60,80,100,120)


class TestTinhBienDoTungChuKy:
    def test_computes_expected_amplitude_sequence(self):
        df = _make_vcp_df()
        diem = tim_dinh_day_cuc_bo(df, khoang_cach_toi_thieu=3)
        chu_ky = tinh_bien_do_tung_chu_ky(diem)
        bien_do = [c["bien_do_pct"] for c in chu_ky]
        # Đã dò số liệu thật — đúng 6 chu kỳ, biên độ giảm dần đều.
        assert bien_do == [22.16, 15.62, 12.4, 10.35, 8.23, 5.21]


class TestXacNhanChuoiCoHep:
    def test_confirms_contraction_over_last_3_cycles(self):
        df = _make_vcp_df()
        diem = tim_dinh_day_cuc_bo(df, khoang_cach_toi_thieu=3)
        chu_ky = tinh_bien_do_tung_chu_ky(diem)
        ket_qua = xac_nhan_chuoi_co_hep(chu_ky, so_chu_ky_toi_thieu=3, dung_sai_pct=3.0)
        assert ket_qua["hop_le"] is True
        assert ket_qua["chuoi_bien_do"] == [10.35, 8.23, 5.21]
        assert ket_qua["ty_le_giam_tong_the"] == pytest.approx(49.7, abs=0.1)

    def test_confirms_contraction_over_all_6_cycles(self):
        df = _make_vcp_df()
        diem = tim_dinh_day_cuc_bo(df, khoang_cach_toi_thieu=3)
        chu_ky = tinh_bien_do_tung_chu_ky(diem)
        ket_qua = xac_nhan_chuoi_co_hep(chu_ky, so_chu_ky_toi_thieu=6, dung_sai_pct=3.0)
        assert ket_qua["hop_le"] is True
        assert ket_qua["so_chu_ky_da_xet"] == 6

    def test_invalid_when_not_enough_cycles(self):
        ket_qua = xac_nhan_chuoi_co_hep([{"bien_do_pct": 10.0}], so_chu_ky_toi_thieu=3)
        assert ket_qua["hop_le"] is False
        assert "cần tối thiểu" in ket_qua["ly_do"]

    def test_invalid_when_amplitude_increases(self):
        chu_ky = [
            {"bien_do_pct": 5.0}, {"bien_do_pct": 8.0}, {"bien_do_pct": 15.0},
        ]
        ket_qua = xac_nhan_chuoi_co_hep(chu_ky, so_chu_ky_toi_thieu=3, dung_sai_pct=1.0)
        assert ket_qua["hop_le"] is False


class TestGanNhanBacBienDo:
    def test_default_thresholds(self):
        assert gan_nhan_bac_bien_do(25.0) == ">20%"
        assert gan_nhan_bac_bien_do(16.0) == "~15%"
        assert gan_nhan_bac_bien_do(4.0) == "~3%"
        assert gan_nhan_bac_bien_do(1.0) == "<3%"

    def test_custom_thresholds_per_symbol(self):
        # BTC biến động mạnh hơn -> bộ ngưỡng riêng cao hơn (mục 10.3 tài liệu gốc).
        nguong_btc = [40.0, 30.0, 20.0, 10.0]
        assert gan_nhan_bac_bien_do(45.0, nguong_btc) == ">40%"
        assert gan_nhan_bac_bien_do(15.0, nguong_btc) == "~10%"


class TestDoiChieuVoiMa20:
    def test_price_above_ma20(self):
        closes = [100.0] * 19 + [110.0]
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=20, freq="D"),
            "open": closes, "high": closes, "low": closes, "close": closes,
            "volume": [1000.0] * 20,
        })
        result = doi_chieu_voi_ma20(df)
        assert bool(result["gia_tren_ma20"]) is True

    def test_returns_none_when_not_enough_data_for_ma20(self):
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "open": [100.0] * 5, "high": [100.0] * 5, "low": [100.0] * 5,
            "close": [100.0] * 5, "volume": [1000.0] * 5,
        })
        result = doi_chieu_voi_ma20(df)
        assert result["gia_tren_ma20"] is None


class TestTuChonKhungThoiGian:
    def test_selects_valid_timeframe(self):
        df = _make_vcp_df()

        def lay_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
            return df

        ket_qua = tu_chon_khung_thoi_gian(
            "XAUUSD", lay_ohlcv, ["1d"], so_ngay_tham_chieu=121, so_chu_ky_toi_thieu=3,
        )
        assert ket_qua["khung_da_chon"] == "1d"
        assert ket_qua["hop_le"] is True

    def test_falls_back_when_no_timeframe_valid(self):
        # Chuỗi giá TĂNG ĐƠN ĐIỆU liên tục (không có điểm đảo chiều nào)
        # -> không tìm được đủ đỉnh/đáy xen kẽ để tạo chu kỳ.
        n = 60
        closes = [100.0 + i * 0.5 for i in range(n)]
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [1000.0] * n,
        })

        def lay_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
            return df

        ket_qua = tu_chon_khung_thoi_gian(
            "XAUUSD", lay_ohlcv, ["1d"], so_ngay_tham_chieu=60, so_chu_ky_toi_thieu=3,
        )
        assert "canh_bao" in ket_qua


class TestRaoSoatMoHinhCoHep:
    def test_end_to_end_pipeline(self):
        df = _make_vcp_df()

        def lay_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
            return df

        result = rao_soat_mo_hinh_co_hep(
            "XAUUSD", lay_ohlcv,
            khung_thoi_gian_ung_vien=("1d",), so_ngay_tham_chieu=121, so_chu_ky_toi_thieu=3,
        )
        assert result["symbol"] == "XAUUSD"
        assert result["xac_nhan_co_hep"] is True
        assert result["chuoi_bien_do_pct"] == [10.35, 8.23, 5.21]
        assert result["chuoi_bac_bien_do"] == ["~10%", "~5%", "~5%"]
        assert "canh_bao_phap_ly" in result and "KHÔNG phải" in result["canh_bao_phap_ly"]

    def test_uses_custom_threshold_per_symbol(self):
        df = _make_vcp_df()

        def lay_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
            return df

        result = rao_soat_mo_hinh_co_hep(
            "BTCUSD", lay_ohlcv,
            khung_thoi_gian_ung_vien=("1d",), so_ngay_tham_chieu=121, so_chu_ky_toi_thieu=3,
            nguong_bac_bien_do=[40.0, 30.0, 20.0, 10.0],
        )
        assert result["chuoi_bac_bien_do"] == ["~10%", "<10%", "<10%"]
