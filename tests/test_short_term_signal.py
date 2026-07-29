"""
Unit test cho core/short_term_signal.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.short_term_signal import (
    NGANH_UU_TIEN_BAT_CA_HOI,
    RO_MA_UU_TIEN_BAT_CA_HOI,
    InvalidShortTermSignalError,
    build_short_term_signal_report,
    canh_bao_qua_mua_co_phieu,
    canh_bao_qua_mua_vnindex,
    kiem_tra_tin_hieu_bat_ca_hoi,
    thong_ke_xac_suat_dieu_chinh,
)


# ==============================================================================
# Test: canh_bao_qua_mua_vnindex
# ==============================================================================

class TestCanhBaoQuaMuaVnindex:
    def test_binh_thuong_duoi_2pct(self):
        result = canh_bao_qua_mua_vnindex(gia_dong_cua=101, ma20=100)
        assert result["do_lech_ma20_pct"] == pytest.approx(1.0)
        assert result["muc_canh_bao"] == "BINH_THUONG"

    def test_canh_bao_dieu_chinh_trong_khoang_2_3pct(self):
        result = canh_bao_qua_mua_vnindex(gia_dong_cua=102.5, ma20=100)
        assert result["muc_canh_bao"] == "CANH_BAO_DIEU_CHINH"

    def test_nguy_co_cao_tu_4pct(self):
        result = canh_bao_qua_mua_vnindex(gia_dong_cua=104, ma20=100)
        assert result["muc_canh_bao"] == "NGUY_CO_CAO"

    def test_bien_dung_2pct_la_canh_bao(self):
        result = canh_bao_qua_mua_vnindex(gia_dong_cua=102, ma20=100)
        assert result["muc_canh_bao"] == "CANH_BAO_DIEU_CHINH"

    def test_bien_dung_4pct_la_nguy_co_cao(self):
        result = canh_bao_qua_mua_vnindex(gia_dong_cua=104, ma20=100)
        assert result["muc_canh_bao"] == "NGUY_CO_CAO"

    def test_raises_for_non_positive_ma20(self):
        with pytest.raises(InvalidShortTermSignalError):
            canh_bao_qua_mua_vnindex(gia_dong_cua=100, ma20=0)


# ==============================================================================
# Test: canh_bao_qua_mua_co_phieu
# ==============================================================================

class TestCanhBaoQuaMuaCoPhieu:
    def test_binh_thuong_duoi_10pct(self):
        result = canh_bao_qua_mua_co_phieu("HPG", gia_dong_cua=105, ma20=100)
        assert result["muc_canh_bao"] == "BINH_THUONG"

    def test_nguy_co_dieu_chinh_10_15pct(self):
        result = canh_bao_qua_mua_co_phieu("HPG", gia_dong_cua=112, ma20=100)
        assert result["muc_canh_bao"] == "NGUY_CO_DIEU_CHINH"

    def test_nguy_co_cao_tren_15pct(self):
        result = canh_bao_qua_mua_co_phieu("HPG", gia_dong_cua=118, ma20=100)
        assert result["muc_canh_bao"] == "NGUY_CO_CAO"
        assert result["ma"] == "HPG"

    def test_raises_for_non_positive_ma20(self):
        with pytest.raises(InvalidShortTermSignalError):
            canh_bao_qua_mua_co_phieu("HPG", gia_dong_cua=100, ma20=-1)


# ==============================================================================
# Test: thong_ke_xac_suat_dieu_chinh — kịch bản tính tay cụ thể
# ==============================================================================

class TestThongKeXacSuatDieuChinh:
    def _make_single_event_series(self, length=35):
        """1 sự kiện DUY NHẤT tại idx=10 (giá vọt lên 105, MA20=100 -> độ
        lệch 5% vượt ngưỡng 4%), sau đó giảm về đáy 99 tại idx=14 (offset
        4 phiên) rồi đi ngang — đã tính tay: mức giảm = (105-99)/105*100
        = 5.714%, đủ vượt ngưỡng điều chỉnh 3% -> ghi nhận 1 sự kiện điều
        chỉnh, đúng 100% xác suất với mẫu 1 sự kiện.
        """
        prices = [100.0] * length
        prices[10] = 105.0
        prices[11] = 104.0
        prices[12] = 103.0
        prices[13] = 101.0
        prices[14] = 99.0
        prices[15] = 100.0
        ma20 = pd.Series([100.0] * length)
        return pd.Series(prices), ma20

    def test_known_single_event_probability_and_magnitude(self):
        gia, ma20 = self._make_single_event_series()
        result = thong_ke_xac_suat_dieu_chinh(
            gia, ma20, nguong_canh_bao_pct=4.0,
            cac_khung_ngay=[5, 10], nguong_dieu_chinh_pct=3.0,
        )

        assert result["tong_so_su_kien"] == 1

        khung5 = result["xac_suat_theo_khung_ngay"][5]
        assert khung5["so_su_kien_hop_le"] == 1
        assert khung5["xac_suat_pct"] == pytest.approx(100.0)
        assert khung5["muc_dieu_chinh_tb_pct"] == pytest.approx((105 - 99) / 105 * 100, abs=0.01)
        assert khung5["so_phien_tb_toi_day"] == pytest.approx(4.0)

        khung10 = result["xac_suat_theo_khung_ngay"][10]
        assert khung10["xac_suat_pct"] == pytest.approx(100.0)

    def test_no_events_when_never_crosses_threshold(self):
        gia = pd.Series([100.0] * 30)
        ma20 = pd.Series([100.0] * 30)
        result = thong_ke_xac_suat_dieu_chinh(gia, ma20, nguong_canh_bao_pct=4.0)
        assert result["tong_so_su_kien"] == 0
        for khung in result["xac_suat_theo_khung_ngay"].values():
            assert khung["xac_suat_pct"] is None
            assert khung["so_su_kien_hop_le"] == 0

    def test_no_correction_when_price_stays_flat_after_event(self):
        prices = [100.0] * 30
        prices[10] = 105.0  # vượt ngưỡng nhưng SAU ĐÓ giữ nguyên, không điều chỉnh
        for i in range(11, 30):
            prices[i] = 105.0
        gia = pd.Series(prices)
        ma20 = pd.Series([100.0] * 30)
        result = thong_ke_xac_suat_dieu_chinh(
            gia, ma20, nguong_canh_bao_pct=4.0, cac_khung_ngay=[5], nguong_dieu_chinh_pct=3.0,
        )
        assert result["xac_suat_theo_khung_ngay"][5]["xac_suat_pct"] == pytest.approx(0.0)

    def test_event_near_end_of_series_excluded_from_that_window(self):
        # Sự kiện tại idx cuối -1, không đủ N phiên tương lai -> so_su_kien_hop_le=0 cho khung đó
        prices = [100.0] * 20
        prices[-1] = 105.0
        gia = pd.Series(prices)
        ma20 = pd.Series([100.0] * 20)
        result = thong_ke_xac_suat_dieu_chinh(gia, ma20, nguong_canh_bao_pct=4.0, cac_khung_ngay=[5])
        assert result["tong_so_su_kien"] == 1
        assert result["xac_suat_theo_khung_ngay"][5]["so_su_kien_hop_le"] == 0
        assert result["xac_suat_theo_khung_ngay"][5]["xac_suat_pct"] is None

    def test_raises_for_mismatched_lengths(self):
        with pytest.raises(InvalidShortTermSignalError):
            thong_ke_xac_suat_dieu_chinh(pd.Series([100, 101]), pd.Series([100]), nguong_canh_bao_pct=4.0)

    def test_raises_for_empty_series(self):
        with pytest.raises(InvalidShortTermSignalError):
            thong_ke_xac_suat_dieu_chinh(pd.Series([], dtype=float), pd.Series([], dtype=float), nguong_canh_bao_pct=4.0)


# ==============================================================================
# Test: kiem_tra_tin_hieu_bat_ca_hoi
# ==============================================================================

class TestKiemTraTinHieuBatCaHoi:
    def test_kich_hoat_khi_giam_trong_khoang_10_15pct(self):
        result = kiem_tra_tin_hieu_bat_ca_hoi(gia_hien_tai=88, lich_su_gia_40_phien=[100] * 40)
        assert result["muc_giam_tu_dinh_40_phien_pct"] == pytest.approx(12.0)
        assert result["kich_hoat"] is True
        assert result["nganh_uu_tien"] == NGANH_UU_TIEN_BAT_CA_HOI
        assert result["ro_ma_uu_tien"] == RO_MA_UU_TIEN_BAT_CA_HOI

    def test_khong_kich_hoat_khi_giam_qua_it(self):
        result = kiem_tra_tin_hieu_bat_ca_hoi(gia_hien_tai=95, lich_su_gia_40_phien=[100] * 40)
        assert result["kich_hoat"] is False
        assert result["nganh_uu_tien"] == []

    def test_veto_khi_macro_score_tieu_cuc_manh(self):
        result = kiem_tra_tin_hieu_bat_ca_hoi(
            gia_hien_tai=88, lich_su_gia_40_phien=[100] * 40, macro_score=-1.5,
        )
        assert result["kich_hoat"] is False
        assert len(result["phu_quyet_ly_do"]) == 1
        assert "Macro Score" in result["phu_quyet_ly_do"][0]

    def test_no_veto_when_macro_score_acceptable(self):
        result = kiem_tra_tin_hieu_bat_ca_hoi(
            gia_hien_tai=88, lich_su_gia_40_phien=[100] * 40, macro_score=0.5,
        )
        assert result["kich_hoat"] is True
        assert result["phu_quyet_ly_do"] == []

    def test_veto_khi_giam_vuot_20pct(self):
        result = kiem_tra_tin_hieu_bat_ca_hoi(gia_hien_tai=75, lich_su_gia_40_phien=[100] * 40)
        assert result["muc_giam_tu_dinh_40_phien_pct"] == pytest.approx(25.0)
        assert result["kich_hoat"] is False
        assert any("20%" in ly_do for ly_do in result["phu_quyet_ly_do"])

    def test_raises_for_empty_history(self):
        with pytest.raises(InvalidShortTermSignalError):
            kiem_tra_tin_hieu_bat_ca_hoi(gia_hien_tai=100, lich_su_gia_40_phien=[])


# ==============================================================================
# Test: build_short_term_signal_report
# ==============================================================================

class TestBuildShortTermSignalReport:
    def test_output_structure_matches_spec(self):
        vnindex_history_close = pd.Series([100.0] * 30)
        vnindex_history_ma20 = pd.Series([100.0] * 30)

        report = build_short_term_signal_report(
            vnindex_close=102.5, vnindex_ma20=100,
            vnindex_history_close=vnindex_history_close,
            vnindex_history_ma20=vnindex_history_ma20,
            vnindex_history_40d=[100.0] * 40,
            stock_snapshots=[
                {"ma": "HPG", "close": 118, "ma20": 100},
                {"ma": "VNM", "close": 101, "ma20": 100},
            ],
            danh_gia_date="2026-07-26",
        )

        expected_keys = {
            "ngay_danh_gia", "vnindex", "tin_hieu_bat_ca_hoi",
            "co_phieu_qua_mua", "canh_bao", "ghi_chu",
        }
        assert expected_keys.issubset(report.keys())
        assert report["vnindex"]["muc_canh_bao"] == "CANH_BAO_DIEU_CHINH"

    def test_only_flagged_stocks_included_in_report(self):
        vnindex_history_close = pd.Series([100.0] * 30)
        vnindex_history_ma20 = pd.Series([100.0] * 30)

        report = build_short_term_signal_report(
            vnindex_close=100, vnindex_ma20=100,
            vnindex_history_close=vnindex_history_close,
            vnindex_history_ma20=vnindex_history_ma20,
            vnindex_history_40d=[100.0] * 40,
            stock_snapshots=[
                {"ma": "HPG", "close": 118, "ma20": 100},  # NGUY_CO_CAO -> có trong báo cáo
                {"ma": "VNM", "close": 101, "ma20": 100},  # BINH_THUONG -> KHÔNG có trong báo cáo
            ],
        )
        ma_list = [s["ma"] for s in report["co_phieu_qua_mua"]]
        assert "HPG" in ma_list
        assert "VNM" not in ma_list

    def test_canh_bao_includes_high_risk_stock_warning(self):
        vnindex_history_close = pd.Series([100.0] * 30)
        vnindex_history_ma20 = pd.Series([100.0] * 30)

        report = build_short_term_signal_report(
            vnindex_close=100, vnindex_ma20=100,
            vnindex_history_close=vnindex_history_close,
            vnindex_history_ma20=vnindex_history_ma20,
            vnindex_history_40d=[100.0] * 40,
            stock_snapshots=[{"ma": "HPG", "close": 120, "ma20": 100}],
        )
        assert any("HPG" in w for w in report["canh_bao"])
