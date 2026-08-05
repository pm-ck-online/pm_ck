"""
Unit test cho core/entry_screener.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.entry_screener import (
    TIEU_CHI_KHA_DUNG,
    InvalidEntryScreenerError,
    detect_bullish_divergence_3_diem,
    kiem_tra_sap_breakout,
    kiem_tra_tich_luy_dai_han,
    quet_danh_sach_cho,
    quet_mot_ma,
    so_sanh_2_khoang_do_lech,
    so_sanh_2_khoang_rsi,
    tinh_kelly_fraction,
    tinh_thong_ke_tang_giam_lich_su,
    xep_hang_uu_tien_theo_duong_tham_chieu,
    xep_hang_uu_tien_theo_ema200,
)


def _make_df(closes, highs=None, lows=None, volumes=None):
    n = len(closes)
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n),
        "open": closes,
        "high": highs or [c * 1.01 for c in closes],
        "low": lows or [c * 0.99 for c in closes],
        "close": closes,
        "volume": volumes or [1000] * n,
    })


# ==============================================================================
# Test: xep_hang_uu_tien_theo_ema200
# ==============================================================================

class TestXepHangUuTienTheoEma200:
    def test_uu_tien_cao_khi_tren_ema200(self):
        result = xep_hang_uu_tien_theo_ema200(gia_dong_cua=105, ema200=100)
        assert result["xep_hang_uu_tien"] == "UU_TIEN_CAO"
        assert result["do_lech_ema200_pct"] == pytest.approx(5.0)

    def test_uu_tien_trung_binh_trong_khoang_am_10pct(self):
        result = xep_hang_uu_tien_theo_ema200(gia_dong_cua=95, ema200=100)
        assert result["xep_hang_uu_tien"] == "UU_TIEN_TRUNG_BINH"

    def test_khong_dat_khi_qua_xa_duoi_ema200(self):
        result = xep_hang_uu_tien_theo_ema200(gia_dong_cua=85, ema200=100)
        assert result["xep_hang_uu_tien"] == "KHONG_DAT"

    def test_bien_dung_am_10pct_van_la_trung_binh(self):
        result = xep_hang_uu_tien_theo_ema200(gia_dong_cua=90, ema200=100)
        assert result["xep_hang_uu_tien"] == "UU_TIEN_TRUNG_BINH"

    def test_ema200_none_tra_ve_khong_dat_khong_loi(self):
        result = xep_hang_uu_tien_theo_ema200(gia_dong_cua=100, ema200=None)
        assert result["xep_hang_uu_tien"] == "KHONG_DAT"
        assert result["do_lech_ema200_pct"] is None

    def test_raises_for_non_positive_ema200(self):
        with pytest.raises(InvalidEntryScreenerError):
            xep_hang_uu_tien_theo_ema200(gia_dong_cua=100, ema200=0)


# ==============================================================================
# Test: kiem_tra_tich_luy_dai_han
# ==============================================================================

class TestKiemTraTichLuyDaiHan:
    def test_dat_khi_bien_do_hep(self):
        df = _make_df(closes=[100.0] * 65, highs=[101.0] * 65, lows=[99.0] * 65)
        result = kiem_tra_tich_luy_dai_han(df, lookback=60, max_range_pct=5.0)
        assert result["dat"] is True

    def test_khong_dat_khi_bien_do_rong(self):
        closes = [100 + (i % 20) for i in range(65)]  # dao động rộng
        df = _make_df(closes)
        result = kiem_tra_tich_luy_dai_han(df, lookback=60, max_range_pct=5.0)
        assert result["dat"] is False

    def test_khong_dat_khi_chua_du_phien(self):
        df = _make_df(closes=[100.0] * 30)
        result = kiem_tra_tich_luy_dai_han(df, lookback=60)
        assert result["dat"] is False
        assert "Chưa đủ" in result["ly_do"]


# ==============================================================================
# Test: kiem_tra_sap_breakout
# ==============================================================================

class TestKiemTraSapBreakout:
    def test_sap_breakout_khi_du_doan_va_bien_do_cuoi_hep(self):
        pattern_result = {
            "segments": [
                {"amplitude_pct": 20.0},
                {"amplitude_pct": 10.0},
                {"amplitude_pct": 4.5},
            ]
        }
        result = kiem_tra_sap_breakout(pattern_result)
        assert result["sap_breakout"] is True
        assert result["bien_do_doan_cuoi_pct"] == pytest.approx(4.5)

    def test_khong_sap_breakout_khi_doan_cuoi_con_rong(self):
        pattern_result = {
            "segments": [
                {"amplitude_pct": 20.0},
                {"amplitude_pct": 10.0},
                {"amplitude_pct": 8.0},  # vẫn còn rộng hơn 5%
            ]
        }
        result = kiem_tra_sap_breakout(pattern_result)
        assert result["sap_breakout"] is False

    def test_khong_sap_breakout_khi_khong_du_so_doan(self):
        pattern_result = {"segments": [{"amplitude_pct": 20.0}, {"amplitude_pct": 4.0}]}
        result = kiem_tra_sap_breakout(pattern_result, so_doan_toi_thieu=3)
        assert result["sap_breakout"] is False
        assert "Chưa đủ" in result["ly_do"]

    def test_none_pattern_result(self):
        result = kiem_tra_sap_breakout(None)
        assert result["sap_breakout"] is False


# ==============================================================================
# Test: detect_bullish_divergence_3_diem — dò số liệu thật
# ==============================================================================

class TestDetectBullishDivergence3Diem:
    def _make_3_point_divergence_df(self):
        """Dữ liệu ĐÃ DÒ KIỂM THỰC TẾ: 3 đáy liên tiếp — đáy 1 (giá=179.1,
        RSI=0.0) -> đáy 2 (giá=163.2 THẤP HƠN, RSI=24.2 CAO HƠN) -> đáy 3
        (giá=156.2 THẤP HƠN NỮA, RSI=29.5 CAO HƠN NỮA) -> phân kỳ liên
        tục qua CẢ 2 cặp đáy, đúng điều kiện "CAO".
        """
        preamble = [300] * 15
        leg0 = [300 - i * 12 for i in range(10)]
        bounce0 = [180 + i * 9 for i in range(6)]
        pattern1 = [-7, -6, 5, -7, -6, 4, -7, -5, 4, -7, -6, 3, -7, -5, 3, -7, -6, 3, -7, -9]
        leg1 = []
        price = 234
        for d in pattern1:
            price += d
            leg1.append(price)
        bounce1 = [price + i * 8 for i in range(6)]
        pattern2 = [-5, -4, 4, -5, -4, 3, -5, -3, 3, -5, -4, 2, -5, -3, 2, -5, -4, 2, -5, -6]
        leg2 = []
        price2 = bounce1[-1]
        for d in pattern2:
            price2 += d
            leg2.append(price2)
        after = [price2 + 5, price2 + 3, price2 + 6, price2 + 4]

        closes = preamble + leg0 + bounce0 + leg1 + bounce1 + leg2 + after
        return pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=len(closes)),
            "open": closes,
            "high": [c * 1.005 for c in closes],
            "low": [c * 0.995 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        })

    def test_detects_3_point_divergence_as_cao(self):
        df = self._make_3_point_divergence_df()
        result = detect_bullish_divergence_3_diem(df, rsi_period=14, lookback=150, swing_order=3)
        assert result["detected"] is True
        assert result["do_tin_cay"] == "CAO"
        assert result["price_low_0"] > result["price_low_1"] > result["price_low_2"]
        assert result["rsi_low_0"] < result["rsi_low_1"] < result["rsi_low_2"]

    def test_2_point_only_yields_trung_binh(self):
        # Dùng lại đúng dữ liệu phân kỳ 2 điểm đã kiểm chứng trước đây
        # (test_market_breadth.py) — không có đáy thứ 3 hợp lệ trước đó.
        preamble = [250, 252, 248, 251, 249, 253, 247, 250, 252, 248, 251, 249, 253, 250, 252]
        leg1 = [250 - i * 10 for i in range(10)]
        bounce = [160 + i * 8 for i in range(6)]
        pattern = [-6, -5, 4, -6, -5, 3, -6, -4, 3, -6, -5, 2, -6, -4, 2, -6, -5, 2, -6, -8]
        leg2 = []
        price = 208
        for d in pattern:
            price += d
            leg2.append(price)
        after = [price + 5, price + 3, price + 6, price + 4]
        closes = preamble + leg1 + bounce + leg2 + after
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=len(closes)),
            "open": closes, "high": [c * 1.005 for c in closes],
            "low": [c * 0.995 for c in closes], "close": closes,
            "volume": [1000] * len(closes),
        })
        result = detect_bullish_divergence_3_diem(df, rsi_period=14, lookback=90, swing_order=3)
        assert result["detected"] is True
        assert result["do_tin_cay"] == "TRUNG_BINH"

    def test_no_divergence_for_monotonic_decline(self):
        n = 60
        closes = [200 - i * 2 for i in range(n)]
        df = _make_df(closes)
        result = detect_bullish_divergence_3_diem(df, rsi_period=14, lookback=60, swing_order=3)
        assert result["detected"] is False

    def test_insufficient_data_returns_reason(self):
        df = _make_df(closes=[100] * 10)
        result = detect_bullish_divergence_3_diem(df)
        assert result["detected"] is False
        assert "reason" in result


# ==============================================================================
# Test: quet_mot_ma / quet_danh_sach_cho
# ==============================================================================

class TestQuetMotMa:
    def test_dat_tieu_chi_ema200(self):
        df = _make_df(closes=[105.0] * 65)
        result = quet_mot_ma(
            "HPG", df, ema200=100, pattern_result=None,
            resistance_level=None, volume_ma20=None,
            tieu_chi_da_chon=["dieu_kien_nen_ema200"],
        )
        assert result is not None
        assert "dieu_kien_nen_ema200" in result["tieu_chi_dat"]
        assert result["xep_hang_uu_tien"] == "UU_TIEN_CAO"

    def test_khong_dat_tieu_chi_nao_tra_ve_none(self):
        df = _make_df(closes=[80.0] * 65)  # quá xa dưới EMA200
        result = quet_mot_ma(
            "HPG", df, ema200=100, pattern_result=None,
            resistance_level=None, volume_ma20=None,
            tieu_chi_da_chon=["dieu_kien_nen_ema200"],
        )
        assert result is None

    def test_dat_tieu_chi_tich_luy(self):
        df = _make_df(closes=[100.0] * 65, highs=[101.0] * 65, lows=[99.0] * 65)
        result = quet_mot_ma(
            "HPG", df, ema200=None, pattern_result=None,
            resistance_level=None, volume_ma20=None,
            tieu_chi_da_chon=["tich_luy_dai_han"],
        )
        assert result is not None
        assert "tich_luy_dai_han" in result["tieu_chi_dat"]

    def test_dat_tieu_chi_volume_breakout(self):
        df = _make_df(closes=[100, 110], volumes=[1000, 2000])
        result = quet_mot_ma(
            "HPG", df, ema200=None, pattern_result=None,
            resistance_level=105, volume_ma20=1000,
            tieu_chi_da_chon=["volume_breakout"],
        )
        assert result is not None
        assert result["mau_hinh_kich_hoat"] == "BREAKOUT"


class TestQuetDanhSachCho:
    def test_sap_xep_theo_uu_tien_giam_dan(self):
        danh_sach = [
            {"symbol": "AAA", "df": _make_df([80.0] * 65), "ema200": 100},   # KHONG_DAT
            {"symbol": "BBB", "df": _make_df([105.0] * 65), "ema200": 100},  # UU_TIEN_CAO
            {"symbol": "CCC", "df": _make_df([95.0] * 65), "ema200": 100},   # UU_TIEN_TRUNG_BINH
        ]
        # Bổ sung để AAA cũng đạt ít nhất 1 tiêu chí khác (tích lũy) hòng có mặt trong kết quả
        danh_sach[0]["df"] = _make_df([80.0] * 65, highs=[80.5] * 65, lows=[79.5] * 65)

        report = quet_danh_sach_cho(
            danh_sach, tieu_chi_da_chon=["dieu_kien_nen_ema200", "tich_luy_dai_han"],
        )
        thu_tu = [m["ma"] for m in report["danh_sach_ma"]]
        assert thu_tu.index("BBB") < thu_tu.index("CCC") < thu_tu.index("AAA")

    def test_tong_so_ma_dat_va_da_quet(self):
        danh_sach = [
            {"symbol": "AAA", "df": _make_df([105.0] * 65), "ema200": 100},
            {"symbol": "BBB", "df": _make_df([80.0] * 65), "ema200": 100},
        ]
        report = quet_danh_sach_cho(danh_sach, tieu_chi_da_chon=["dieu_kien_nen_ema200"])
        assert report["tong_so_ma_da_quet"] == 2
        assert report["tong_so_ma_dat"] == 1

    def test_raises_for_empty_criteria(self):
        with pytest.raises(InvalidEntryScreenerError):
            quet_danh_sach_cho([], tieu_chi_da_chon=[])

    def test_raises_for_invalid_criteria(self):
        with pytest.raises(InvalidEntryScreenerError):
            quet_danh_sach_cho([], tieu_chi_da_chon=["khong_ton_tai"])

    def test_all_available_criteria_keys_are_valid(self):
        # Đảm bảo TIEU_CHI_KHA_DUNG không tự mâu thuẫn với logic validate
        for key in TIEU_CHI_KHA_DUNG:
            quet_danh_sach_cho([], tieu_chi_da_chon=[key])  # không raise


class TestTinhThongKeTangGiamLichSu:
    def _make_realistic_df(self, n=400, seed=5):
        import numpy as np
        np.random.seed(seed)
        dates = pd.bdate_range("2024-01-01", periods=n)
        closes = [100.0]
        for i in range(1, n):
            closes.append(closes[-1] * (1 + np.random.normal(0.0003, 0.004)))
        closes = np.array(closes)
        for start in range(250, 350, 30):
            if start + 30 < n:
                closes[start + 30:] *= 1.08
        return pd.DataFrame({
            "date": dates, "open": closes, "high": closes * 1.005, "low": closes * 0.995,
            "close": closes, "volume": np.random.randint(1000, 2000, n).astype(float),
        })

    def test_computes_distribution_for_fast_criteria(self):
        df = self._make_realistic_df()
        result = tinh_thong_ke_tang_giam_lich_su(
            df, ["dieu_kien_nen_ema200"], so_phien_du_bao=30, so_phien_kiem_tra=250,
        )
        assert result["so_lan_quan_sat"] > 0
        assert set(result["phan_bo"].keys()) == {
            "giam_tren_15", "giam_10_15", "giam_5_10", "giam_0_5",
            "tang_0_5", "tang_5_10", "tang_10_15", "tang_tren_15",
        }
        tong_ty_le = sum(v["ty_le_pct"] for v in result["phan_bo"].values())
        assert tong_ty_le == pytest.approx(100.0, abs=0.5)

    def test_returns_zero_when_only_slow_criteria(self):
        df = self._make_realistic_df()
        result = tinh_thong_ke_tang_giam_lich_su(df, ["dao_dong_tat_dan"])
        assert result["so_lan_quan_sat"] == 0
        assert "KHÔNG được phát lại" in result["ghi_chu"]

    def test_returns_zero_when_history_too_short(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=50),
            "close": [100.0] * 50,
        })
        result = tinh_thong_ke_tang_giam_lich_su(df, ["dieu_kien_nen_ema200"])
        assert result["so_lan_quan_sat"] == 0

    def test_returns_zero_for_empty_df(self):
        result = tinh_thong_ke_tang_giam_lich_su(pd.DataFrame(), ["dieu_kien_nen_ema200"])
        assert result["so_lan_quan_sat"] == 0

    def test_combines_both_fast_criteria(self):
        df = self._make_realistic_df()
        result = tinh_thong_ke_tang_giam_lich_su(
            df, ["dieu_kien_nen_ema200", "tich_luy_dai_han"], so_phien_du_bao=30,
        )
        assert result["so_lan_quan_sat"] > 0
        assert "Tích lũy" in result["ghi_chu"] or "EMA200" in result["ghi_chu"]

    def test_works_with_ma20_reference_line(self):
        df = self._make_realistic_df()
        result = tinh_thong_ke_tang_giam_lich_su(
            df, ["dieu_kien_nen_ema200"], so_phien_du_bao=10, duong_tham_chieu="ma20",
        )
        assert result["so_lan_quan_sat"] > 0

    def test_ma20_needs_less_history_than_ema200(self):
        # MA20 chỉ cần tối thiểu 20 phiên, ít hơn nhiều so với 200 của EMA200.
        df = self._make_realistic_df(n=60)
        r_ema200 = tinh_thong_ke_tang_giam_lich_su(
            df, ["dieu_kien_nen_ema200"], so_phien_du_bao=10, duong_tham_chieu="ema200",
        )
        r_ma20 = tinh_thong_ke_tang_giam_lich_su(
            df, ["dieu_kien_nen_ema200"], so_phien_du_bao=10, duong_tham_chieu="ma20",
        )
        assert r_ema200["so_lan_quan_sat"] == 0  # chưa đủ 200 phiên cho EMA200
        assert r_ma20["so_lan_quan_sat"] > 0      # đã đủ 20 phiên cho MA20

    def test_raises_for_invalid_duong_tham_chieu(self):
        df = self._make_realistic_df()
        with pytest.raises(InvalidEntryScreenerError):
            tinh_thong_ke_tang_giam_lich_su(df, ["dieu_kien_nen_ema200"], duong_tham_chieu="ma50")


class TestXepHangUuTienTheoDuongThamChieu:
    def test_matches_ema200_wrapper_exactly(self):
        # Hàm cũ (wrapper) và hàm tổng quát PHẢI cho kết quả giống hệt nhau.
        ket_qua_cu = xep_hang_uu_tien_theo_ema200(105.0, 100.0)
        ket_qua_moi = xep_hang_uu_tien_theo_duong_tham_chieu(105.0, 100.0, "EMA200")
        assert ket_qua_cu["xep_hang_uu_tien"] == ket_qua_moi["xep_hang_uu_tien"]
        assert ket_qua_cu["do_lech_ema200_pct"] == pytest.approx(ket_qua_moi["do_lech_pct"])

    def test_works_with_ma20_label(self):
        ket_qua = xep_hang_uu_tien_theo_duong_tham_chieu(110.0, 100.0, "MA20")
        assert ket_qua["xep_hang_uu_tien"] == "UU_TIEN_CAO"
        assert ket_qua["do_lech_pct"] == pytest.approx(10.0)

    def test_none_reference_returns_khong_dat(self):
        ket_qua = xep_hang_uu_tien_theo_duong_tham_chieu(100.0, None, "MA20")
        assert ket_qua["xep_hang_uu_tien"] == "KHONG_DAT"
        assert ket_qua["do_lech_pct"] is None

    def test_raises_for_non_positive_reference(self):
        with pytest.raises(InvalidEntryScreenerError):
            xep_hang_uu_tien_theo_duong_tham_chieu(100.0, 0.0, "MA20")


class TestTinhKellyFraction:
    def _phan_bo_gia_lap(self, ty_le_theo_bac: dict, gia_tri_theo_bac: dict, tong: int = 100):
        phan_bo = {}
        for key in [
            "giam_tren_15", "giam_10_15", "giam_5_10", "giam_0_5",
            "tang_0_5", "tang_5_10", "tang_10_15", "tang_tren_15",
        ]:
            so_lan = round(tong * ty_le_theo_bac.get(key, 0) / 100)
            phan_bo[key] = {
                "so_lan": so_lan,
                "gia_tri_trung_binh_pct": gia_tri_theo_bac.get(key),
            }
        return phan_bo

    def test_positive_edge_gives_positive_kelly(self):
        # Xác suất thắng cao (70%) + biên độ thắng/thua tương đương -> Kelly dương rõ ràng.
        phan_bo = self._phan_bo_gia_lap(
            {"tang_0_5": 70.0, "giam_0_5": 30.0},
            {"tang_0_5": 5.0, "giam_0_5": -5.0},
        )
        ket_qua = tinh_kelly_fraction(phan_bo)
        assert ket_qua["kelly_f"] > 0
        assert ket_qua["kelly_f_nua"] == pytest.approx(ket_qua["kelly_f"] / 2)

    def test_no_edge_gives_zero_kelly(self):
        # 50/50 với biên độ thắng/thua bằng nhau -> không có lợi thế -> Kelly = 0.
        phan_bo = self._phan_bo_gia_lap(
            {"tang_0_5": 50.0, "giam_0_5": 50.0},
            {"tang_0_5": 5.0, "giam_0_5": -5.0},
        )
        ket_qua = tinh_kelly_fraction(phan_bo)
        assert ket_qua["kelly_f"] == 0.0
        assert "không nên vào lệnh" in ket_qua["ghi_chu"]

    def test_negative_edge_capped_at_zero(self):
        # Xác suất thắng thấp + biên độ thua LỚN hơn thắng -> Kelly âm -> cắt về 0.
        phan_bo = self._phan_bo_gia_lap(
            {"tang_0_5": 30.0, "giam_0_5": 70.0},
            {"tang_0_5": 3.0, "giam_0_5": -10.0},
        )
        ket_qua = tinh_kelly_fraction(phan_bo)
        assert ket_qua["kelly_f"] == 0.0
        assert ket_qua["kelly_f_tho"] < 0  # giá trị thô âm, xác nhận không phải trùng hợp

    def test_kelly_capped_at_100_percent(self):
        # Lợi thế cực lớn -> f* thô có thể > 1 -> phải cắt về đúng 1.0 (100%).
        phan_bo = self._phan_bo_gia_lap(
            {"tang_0_5": 90.0, "giam_0_5": 10.0},
            {"tang_0_5": 20.0, "giam_0_5": -2.0},
        )
        ket_qua = tinh_kelly_fraction(phan_bo)
        assert ket_qua["kelly_f"] == 1.0

    def test_returns_none_when_no_observations(self):
        phan_bo = self._phan_bo_gia_lap({}, {})
        ket_qua = tinh_kelly_fraction(phan_bo)
        assert ket_qua["kelly_f"] is None

    def test_returns_none_when_only_one_direction(self):
        # Chỉ toàn "tăng", không có lần nào "giảm" -> không tính được Kelly (thiếu L).
        phan_bo = self._phan_bo_gia_lap(
            {"tang_0_5": 100.0}, {"tang_0_5": 5.0},
        )
        ket_qua = tinh_kelly_fraction(phan_bo)
        assert ket_qua["kelly_f"] is None
        assert "thiếu quan sát" in ket_qua["ghi_chu"]


class TestSoSanh2KhoangDoLech:
    def _make_engineered_df(self, n=1000, seed=11):
        import numpy as np
        from core.indicators import calculate_ma

        np.random.seed(seed)
        dates = pd.bdate_range("2021-01-01", periods=n)
        closes = 100 + np.cumsum(np.random.normal(0, 1.2, n))
        df = pd.DataFrame({
            "date": dates, "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": np.full(n, 1_000_000.0),
        })

        ma20_series = calculate_ma(df, period=20)
        final_closes = closes.copy()
        for i in range(20, n - 5):
            ma20_i = ma20_series.iloc[i]
            if pd_isna_safe(ma20_i) or ma20_i <= 0:
                continue
            do_lech = (closes[i] - ma20_i) / ma20_i * 100  # CÓ DẤU, khớp với logic hàm mới
            if 5 <= do_lech < 10:
                final_closes[i + 5] = closes[i] * 1.15   # hiệu ứng MẠNH, rõ ràng
            elif 0 <= do_lech < 5:
                final_closes[i + 5] = closes[i] * 0.999  # gần như đi ngang, hơi giảm nhẹ

        df["close"] = final_closes
        df["high"] = final_closes * 1.01
        df["low"] = final_closes * 0.99
        return df

    def test_detects_significant_difference_when_engineered(self):
        df = self._make_engineered_df()
        ket_qua = so_sanh_2_khoang_do_lech(
            df, khoang_1=(0, 5), khoang_2=(5, 10), duong_tham_chieu="ma20",
            so_phien_du_bao=5, so_phien_kiem_tra=1000,
        )
        assert ket_qua["hop_le"] is True
        assert ket_qua["co_y_nghia_thong_ke"] is True
        assert ket_qua["p_value"] < 0.05
        assert ket_qua["xac_suat_thang_khoang_2_pct"] > ket_qua["xac_suat_thang_khoang_1_pct"]

    def test_no_significant_difference_on_pure_random_walk(self):
        import numpy as np
        np.random.seed(42)
        n = 500
        closes = 100 + np.cumsum(np.random.normal(0.02, 1.0, n))
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=n),
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": np.random.randint(500_000, 2_000_000, n).astype(float),
        })
        ket_qua = so_sanh_2_khoang_do_lech(
            df, khoang_1=(0, 5), khoang_2=(5, 10), duong_tham_chieu="ma20", so_phien_du_bao=5,
        )
        # Không có logic nào tạo khác biệt thật -> mong đợi KHÔNG có ý nghĩa thống kê.
        assert ket_qua["co_y_nghia_thong_ke"] is False

    def test_returns_invalid_when_sample_too_small(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=30),
            "open": [100.0] * 30, "high": [100.0] * 30, "low": [100.0] * 30,
            "close": [100.0] * 30, "volume": [1_000_000.0] * 30,
        })
        ket_qua = so_sanh_2_khoang_do_lech(df, khoang_1=(0, 5), khoang_2=(5, 10), duong_tham_chieu="ma20")
        assert ket_qua["hop_le"] is False

    def test_raises_for_invalid_duong_tham_chieu(self):
        df = self._make_engineered_df()
        with pytest.raises(InvalidEntryScreenerError):
            so_sanh_2_khoang_do_lech(df, khoang_1=(0, 5), khoang_2=(5, 10), duong_tham_chieu="ma50")

    def test_result_types_are_json_serializable(self):
        import json
        df = self._make_engineered_df()
        ket_qua = so_sanh_2_khoang_do_lech(
            df, khoang_1=(0, 5), khoang_2=(5, 10), duong_tham_chieu="ma20",
            so_phien_du_bao=5, so_phien_kiem_tra=1000,
        )
        json.dumps(ket_qua)  # không được ném lỗi TypeError (numpy bool/float)

    def test_tong_so_phien_quet_included_and_ge_tong_2_khoang(self):
        df = self._make_engineered_df()
        ket_qua = so_sanh_2_khoang_do_lech(
            df, khoang_1=(0, 5), khoang_2=(5, 10), duong_tham_chieu="ma20",
            so_phien_du_bao=5, so_phien_kiem_tra=1000,
        )
        assert "tong_so_phien_quet" in ket_qua
        assert ket_qua["tong_so_phien_quet"] >= ket_qua["so_lan_khoang_1"] + ket_qua["so_lan_khoang_2"]

    def test_supports_negative_range_below_reference_line(self):
        # Khoảng ÂM (dưới đường tham chiếu) — đúng yêu cầu bổ sung: so
        # sánh -5% đến 0% (dưới MA20) với 0% đến +5% (trên MA20).
        df = self._make_engineered_df()
        ket_qua = so_sanh_2_khoang_do_lech(
            df, khoang_1=(-5, 0), khoang_2=(0, 5), duong_tham_chieu="ma20",
            so_phien_du_bao=5, so_phien_kiem_tra=1000,
        )
        assert ket_qua["hop_le"] is True
        assert ket_qua["so_lan_khoang_1"] > 0  # phải bắt được các phiên có độ lệch ÂM

    def test_raises_when_tu_not_less_than_den(self):
        df = self._make_engineered_df()
        with pytest.raises(InvalidEntryScreenerError):
            so_sanh_2_khoang_do_lech(df, khoang_1=(5, 5), khoang_2=(5, 10), duong_tham_chieu="ma20")
        with pytest.raises(InvalidEntryScreenerError):
            so_sanh_2_khoang_do_lech(df, khoang_1=(0, 5), khoang_2=(10, 5), duong_tham_chieu="ma20")


def pd_isna_safe(v):
    import pandas as pd
    return pd.isna(v)


class TestSoSanh2KhoangRsi:
    def _make_df(self, n=500, seed=5):
        import numpy as np
        np.random.seed(seed)
        dates = pd.bdate_range("2023-01-01", periods=n)
        closes = 100 + np.cumsum(np.random.normal(0.02, 1.0, n))
        return pd.DataFrame({
            "date": dates, "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": np.random.randint(500_000, 2_000_000, n).astype(float),
        })

    def test_computes_valid_result_for_oversold_vs_overbought(self):
        df = self._make_df()
        ket_qua = so_sanh_2_khoang_rsi(df, khoang_1=(0, 30), khoang_2=(70, 101), so_phien_du_bao=5)
        assert ket_qua["hop_le"] is True
        assert ket_qua["so_lan_khoang_1"] >= 5
        assert ket_qua["so_lan_khoang_2"] >= 5
        assert 0 <= ket_qua["xac_suat_thang_khoang_1_pct"] <= 100
        assert 0 <= ket_qua["xac_suat_thang_khoang_2_pct"] <= 100

    def test_three_canonical_zones_dont_overlap(self):
        # Quá bán (0,30), Trung tính (30,70), Quá mua (70,101) — không chồng lấn.
        qua_ban, trung_tinh, qua_mua = (0, 30), (30, 70), (70, 101)
        assert qua_ban[1] == trung_tinh[0]
        assert trung_tinh[1] == qua_mua[0]

    def test_raises_when_tu_not_less_than_den(self):
        df = self._make_df()
        with pytest.raises(InvalidEntryScreenerError):
            so_sanh_2_khoang_rsi(df, khoang_1=(30, 30), khoang_2=(70, 101))
        with pytest.raises(InvalidEntryScreenerError):
            so_sanh_2_khoang_rsi(df, khoang_1=(0, 30), khoang_2=(101, 70))

    def test_returns_invalid_when_history_too_short(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=10),
            "open": [100.0] * 10, "high": [100.0] * 10, "low": [100.0] * 10,
            "close": [100.0] * 10, "volume": [1_000_000.0] * 10,
        })
        ket_qua = so_sanh_2_khoang_rsi(df, khoang_1=(0, 30), khoang_2=(70, 101))
        assert ket_qua["hop_le"] is False

    def test_result_is_json_serializable(self):
        import json
        df = self._make_df()
        ket_qua = so_sanh_2_khoang_rsi(df, khoang_1=(0, 30), khoang_2=(70, 101), so_phien_du_bao=5)
        json.dumps(ket_qua)

    def test_needs_less_history_than_ma_based_test(self):
        # RSI(14) cần ít dữ liệu hơn nhiều so với EMA200 (200 phiên).
        df = self._make_df(n=60)
        ket_qua = so_sanh_2_khoang_rsi(df, khoang_1=(0, 30), khoang_2=(70, 101), so_phien_du_bao=5)
        # Không bắt buộc phải hợp lệ (còn tùy cỡ mẫu thực tế rơi vào mỗi
        # khoảng), nhưng KHÔNG được báo "chưa đủ dữ liệu" vì đã đủ 60 phiên.
        if not ket_qua["hop_le"]:
            assert "Chưa đủ dữ liệu lịch sử" not in ket_qua.get("ghi_chu", "")

    def test_tong_so_phien_quet_included_and_ge_tong_2_khoang(self):
        df = self._make_df()
        ket_qua = so_sanh_2_khoang_rsi(df, khoang_1=(0, 30), khoang_2=(70, 101), so_phien_du_bao=5)
        assert "tong_so_phien_quet" in ket_qua
        assert ket_qua["tong_so_phien_quet"] >= ket_qua["so_lan_khoang_1"] + ket_qua["so_lan_khoang_2"]
