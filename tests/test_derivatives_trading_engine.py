"""Unit test cho core/derivatives_trading_engine.py.

Giá trị kỳ vọng đã được TỰ TÍNH LẠI VÀ XÁC NHẬN đúng công thức trước khi
viết test (không chép nguyên số liệu minh họa trong tài liệu gốc — đã
phát hiện 1 điểm không nhất quán trong ví dụ minh họa gốc, xem giải
thích trong core/derivatives_trading_engine.py).
"""

from __future__ import annotations

import pytest

from core.derivatives_trading_engine import (
    HE_SO_BIEN_DO_ENTRY,
    HUONG_THEO_TIN_HIEU,
    InvalidDerivativesError,
    phan_tich_lenh_hdtl_vn30,
    tinh_khoang_gia_vao_lenh,
    tinh_so_hop_dong_toi_uu,
    tinh_ty_le_rr,
)


class TestTinhKhoangGiaVaoLenh:
    def test_mua_ho_tro(self):
        ket_qua = tinh_khoang_gia_vao_lenh(1830.0, 25.0, "MUA_HO_TRO")
        assert ket_qua == (1825.0, 1835.0)

    def test_breakout_tang(self):
        ket_qua = tinh_khoang_gia_vao_lenh(1830.0, 25.0, "BREAKOUT_TANG")
        assert ket_qua == (1831.2, 1840.0)

    def test_breakout_giam(self):
        ket_qua = tinh_khoang_gia_vao_lenh(1830.0, 25.0, "BREAKOUT_GIAM")
        assert ket_qua == (1820.0, 1828.8)

    def test_ban_khang_cu_same_offset_as_mua_ho_tro(self):
        # Cùng hệ số biên độ (-0.20, 0.20) như MUA_HO_TRO — khác nhau ở
        # HƯỚNG lệnh (LONG/SHORT), không phải công thức entry.
        assert HE_SO_BIEN_DO_ENTRY["BAN_KHANG_CU"] == HE_SO_BIEN_DO_ENTRY["MUA_HO_TRO"]

    def test_raises_for_invalid_kieu_tin_hieu(self):
        with pytest.raises(InvalidDerivativesError):
            tinh_khoang_gia_vao_lenh(1830.0, 25.0, "KHONG_TON_TAI")

    def test_raises_for_non_positive_gia_tham_chieu(self):
        with pytest.raises(InvalidDerivativesError):
            tinh_khoang_gia_vao_lenh(0.0, 25.0, "MUA_HO_TRO")

    def test_raises_for_non_positive_atr14(self):
        with pytest.raises(InvalidDerivativesError):
            tinh_khoang_gia_vao_lenh(1830.0, 0.0, "MUA_HO_TRO")

    def test_always_returns_ascending_tuple(self):
        # Kể cả BREAKOUT_GIAM (hệ số âm) vẫn phải trả về (thấp, cao) tăng dần.
        thap, cao = tinh_khoang_gia_vao_lenh(1830.0, 25.0, "BREAKOUT_GIAM")
        assert thap < cao


class TestTinhSoHopDongToiUu:
    def test_khop_dung_vi_du_da_xac_nhan(self):
        # Số liệu: NAV 1,5 tỷ, entry LONG tối đa 1835, cắt lỗ 1812.
        ket_qua = tinh_so_hop_dong_toi_uu(
            nav=1_500_000_000, gia_vao_toi_da=1835.0, gia_cat_lo=1812.0,
            gia_tham_chieu_ky_quy=1835.0,
        )
        assert ket_qua["so_hop_dong"] == 13
        assert ket_qua["rui_ro_tren_1_hd_vnd"] == 2_300_000
        assert ket_qua["ky_quy_yeu_cau_1_hd_vnd"] == 27_525_000
        assert ket_qua["tong_ky_quy_su_dung_vnd"] == 357_825_000
        assert ket_qua["tong_ky_quy_pct_nav"] == pytest.approx(23.9, abs=0.05)
        assert ket_qua["nut_that_gioi_han"] == "theo_rui_ro_2pct"
        assert ket_qua["chi_tiet_cac_rang_buoc"]["theo_tran_ky_quy"] == 27

    def test_nut_that_la_tran_ky_quy_khi_cat_lo_hep(self):
        # Cắt lỗ RẤT hẹp (lướt trong ngày) -> rủi ro/1 HĐ nhỏ -> số HĐ theo
        # rủi ro rất lớn -> ràng buộc TRẦN KÝ QUỸ mới là nút thắt.
        ket_qua = tinh_so_hop_dong_toi_uu(
            nav=1_500_000_000, gia_vao_toi_da=1835.0, gia_cat_lo=1834.5,
            gia_tham_chieu_ky_quy=1835.0,
        )
        assert ket_qua["nut_that_gioi_han"] == "theo_tran_ky_quy"

    def test_nut_that_la_gioi_han_quy_dinh_khi_nav_rat_lon(self):
        ket_qua = tinh_so_hop_dong_toi_uu(
            nav=1_000_000_000_000, gia_vao_toi_da=1835.0, gia_cat_lo=1812.0,
            gia_tham_chieu_ky_quy=1835.0,
        )
        assert ket_qua["nut_that_gioi_han"] == "theo_gioi_han_quy_dinh"
        assert ket_qua["so_hop_dong"] == 500

    def test_khong_bao_gio_vuot_tran_ky_quy(self):
        for nav in (100_000_000, 500_000_000, 2_000_000_000, 10_000_000_000):
            ket_qua = tinh_so_hop_dong_toi_uu(
                nav=nav, gia_vao_toi_da=1835.0, gia_cat_lo=1812.0,
                gia_tham_chieu_ky_quy=1835.0, ty_le_ky_quy_toi_da_nav=0.50,
            )
            assert ket_qua["tong_ky_quy_pct_nav"] <= 50.05

    def test_raises_for_non_positive_nav(self):
        with pytest.raises(InvalidDerivativesError):
            tinh_so_hop_dong_toi_uu(nav=0, gia_vao_toi_da=1835.0, gia_cat_lo=1812.0, gia_tham_chieu_ky_quy=1835.0)

    def test_raises_when_gia_vao_equals_gia_cat_lo(self):
        with pytest.raises(InvalidDerivativesError):
            tinh_so_hop_dong_toi_uu(
                nav=1_500_000_000, gia_vao_toi_da=1835.0, gia_cat_lo=1835.0,
                gia_tham_chieu_ky_quy=1835.0,
            )

    def test_raises_for_ty_le_ky_quy_out_of_range(self):
        with pytest.raises(InvalidDerivativesError):
            tinh_so_hop_dong_toi_uu(
                nav=1_500_000_000, gia_vao_toi_da=1835.0, gia_cat_lo=1812.0,
                gia_tham_chieu_ky_quy=1835.0, ty_le_ky_quy=1.5,
            )


class TestTinhTyLeRR:
    def test_long_tot(self):
        ket_qua = tinh_ty_le_rr(1830.0, 1812.0, 1899.0, "LONG")
        assert ket_qua["rr"] == pytest.approx(3.83, abs=0.01)
        assert ket_qua["danh_gia"] == "TOT"

    def test_long_thap_can_xem_xet_lai(self):
        # Rủi ro 18 điểm, lợi nhuận ~20.5 điểm -> R:R ~1.14 (khớp đúng nhận
        # xét Kịch bản 1/mục tiêu gần trong tài liệu gốc).
        ket_qua = tinh_ty_le_rr(1830.0, 1812.0, 1850.5, "LONG")
        assert ket_qua["danh_gia"] == "THAP_CAN_XEM_XET_LAI"

    def test_short(self):
        ket_qua = tinh_ty_le_rr(1830.0, 1848.0, 1794.0, "SHORT")
        assert ket_qua["rr"] == 2.0
        assert ket_qua["danh_gia"] == "TOT"

    def test_nguong_danh_gia_chinh_xac(self):
        assert tinh_ty_le_rr(100.0, 90.0, 300.0, "LONG")["danh_gia"] == "TOT"  # rr=20
        assert tinh_ty_le_rr(100.0, 90.0, 115.0, "LONG")["danh_gia"] == "CHAP_NHAN_DUOC"  # rr=1.5

    def test_khong_nen_vao_lenh(self):
        ket_qua = tinh_ty_le_rr(100.0, 90.0, 105.0, "LONG")
        assert ket_qua["rr"] == 0.5
        assert ket_qua["danh_gia"] == "KHONG_NEN_VAO_LENH"

    def test_raises_for_invalid_huong(self):
        with pytest.raises(InvalidDerivativesError):
            tinh_ty_le_rr(100.0, 90.0, 120.0, "NGANG")

    def test_raises_for_invalid_stop_loss_direction(self):
        # LONG nhưng cắt lỗ đặt CAO HƠN giá vào -> rủi ro âm -> lỗi.
        with pytest.raises(InvalidDerivativesError):
            tinh_ty_le_rr(100.0, 110.0, 120.0, "LONG")


class TestPhanTichLenhHdtlVn30:
    def test_end_to_end_khop_vi_du(self):
        ket_qua = phan_tich_lenh_hdtl_vn30(
            nav=1_500_000_000, phong_cach="giu_theo_kich_ban", kieu_tin_hieu="MUA_HO_TRO",
            gia_tham_chieu=1830.0, atr14=25.0, gia_cat_lo=1812.0, gia_chot_loi_du_kien=1899.0,
        )
        assert ket_qua["huong"] == "LONG"
        assert ket_qua["khoang_gia_vao_lenh"] == (1825.0, 1835.0)
        assert ket_qua["so_hop_dong"] == 13
        assert ket_qua["ty_le_rr"]["danh_gia"] == "TOT"
        assert ket_qua["canh_bao"] == []
        assert "canh_bao_phap_ly" in ket_qua

    def test_short_signal(self):
        ket_qua = phan_tich_lenh_hdtl_vn30(
            nav=1_500_000_000, phong_cach="luot_trong_ngay", kieu_tin_hieu="BREAKOUT_GIAM",
            gia_tham_chieu=1830.0, atr14=10.0, gia_cat_lo=1845.0,
        )
        assert ket_qua["huong"] == "SHORT"
        assert ket_qua["ty_le_rr"] is None  # không truyền gia_chot_loi_du_kien

    def test_canh_bao_khi_rr_thap(self):
        ket_qua = phan_tich_lenh_hdtl_vn30(
            nav=1_500_000_000, phong_cach="giu_theo_kich_ban", kieu_tin_hieu="MUA_HO_TRO",
            gia_tham_chieu=1830.0, atr14=25.0, gia_cat_lo=1812.0, gia_chot_loi_du_kien=1835.0,
        )
        assert len(ket_qua["canh_bao"]) == 1

    def test_raises_for_invalid_phong_cach(self):
        with pytest.raises(InvalidDerivativesError):
            phan_tich_lenh_hdtl_vn30(
                nav=1_500_000_000, phong_cach="khong_hop_le", kieu_tin_hieu="MUA_HO_TRO",
                gia_tham_chieu=1830.0, atr14=25.0, gia_cat_lo=1812.0,
            )

    def test_huong_theo_tin_hieu_day_du(self):
        assert HUONG_THEO_TIN_HIEU == {
            "BREAKOUT_TANG": "LONG", "MUA_HO_TRO": "LONG",
            "BREAKOUT_GIAM": "SHORT", "BAN_KHANG_CU": "SHORT",
        }
