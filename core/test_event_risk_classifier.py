"""
tests/test_event_risk_classifier.py

Unit test cho core/event_risk_classifier.py — bao gồm case thực tế đã phân
tích: xung đột Mỹ-Iran (Nhóm A) xảy ra ĐỒNG THỜI với rủi ro can thiệp tỷ giá
yên Mỹ-Nhật (Nhóm B), ngày 2/8/2026.
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.event_risk_classifier import (
    SuKien,
    tra_diem_su_kien,
    tra_mo_ta_su_kien,
    tinh_diem_su_kien_tong_hop,
    ap_dung_override_macro_score,
    dung_su_kien_tu_dict,
    danh_sach_nhan_muc_do_hop_le,
    NHOM_DIA_CHINH_TRI,
    NHOM_TAI_CHINH_TIEN_TE,
    NHOM_THIEN_TAI,
    NhomSuKienKhongHopLeError,
    MucDoKhongHopLeError,
)


# ---------------------------------------------------------------------------
# Test tra điểm từng sự kiện đơn lẻ
# ---------------------------------------------------------------------------

def test_tra_diem_dia_chinh_tri_xung_dot():
    sk = SuKien(nhom=NHOM_DIA_CHINH_TRI, muc_do="xung_dot_no_ra")
    assert tra_diem_su_kien(sk) == -2.0


def test_tra_diem_tai_chinh_canh_bao_som():
    """Đúng case thực tế: Mỹ-Nhật can thiệp cứu yên 2/8/2026."""
    sk = SuKien(
        nhom=NHOM_TAI_CHINH_TIEN_TE,
        muc_do="canh_bao_som",
        ghi_chu="Mỹ-Nhật phối hợp can thiệp cứu yên",
        ngay_ghi_nhan="2026-08-02",
    )
    assert tra_diem_su_kien(sk) == -0.5
    assert "carry-trade" in tra_mo_ta_su_kien(sk)


def test_tra_diem_thien_tai_dien_rong():
    sk = SuKien(nhom=NHOM_THIEN_TAI, muc_do="dien_rong")
    assert tra_diem_su_kien(sk) == -2.0


def test_nhom_khong_hop_le_raise_error():
    with pytest.raises(NhomSuKienKhongHopLeError):
        SuKien(nhom="nhom_khong_ton_tai", muc_do="binh_thuong")


def test_muc_do_khong_hop_le_raise_error():
    with pytest.raises(MucDoKhongHopLeError):
        SuKien(nhom=NHOM_DIA_CHINH_TRI, muc_do="muc_do_khong_ton_tai")


# ---------------------------------------------------------------------------
# Test tổng hợp nhiều sự kiện đồng thời — quy tắc MIN
# ---------------------------------------------------------------------------

def test_tong_hop_rong_tra_ve_0():
    kq = tinh_diem_su_kien_tong_hop([])
    assert kq["score_event"] == 0.0
    assert kq["su_kien_quyet_dinh"] is None
    assert kq["chi_tiet"] == []


def test_tong_hop_1_su_kien():
    sk = SuKien(nhom=NHOM_TAI_CHINH_TIEN_TE, muc_do="canh_bao_som")
    kq = tinh_diem_su_kien_tong_hop([sk])
    assert kq["score_event"] == -0.5
    assert kq["su_kien_quyet_dinh"]["nhom"] == NHOM_TAI_CHINH_TIEN_TE


def test_tong_hop_ca_2_su_kien_dong_thoi_lay_min():
    """
    Case thực tế đã phân tích: xung đột Mỹ-Iran (Nhóm A, -2.0) đang diễn ra
    ĐỒNG THỜI với cảnh báo sớm rủi ro yên (Nhóm B, -0.5).
    -> score_event_tong phải = MIN(-2.0, -0.5) = -2.0 (không phải cộng dồn -2.5).
    """
    sk_iran = SuKien(nhom=NHOM_DIA_CHINH_TRI, muc_do="xung_dot_no_ra", ghi_chu="Mỹ-Iran")
    sk_yen = SuKien(
        nhom=NHOM_TAI_CHINH_TIEN_TE, muc_do="canh_bao_som", ghi_chu="Can thiệp yên Mỹ-Nhật"
    )
    kq = tinh_diem_su_kien_tong_hop([sk_iran, sk_yen])

    assert kq["score_event"] == -2.0  # MIN, không phải -2.5 (cộng dồn)
    assert kq["su_kien_quyet_dinh"]["nhom"] == NHOM_DIA_CHINH_TRI
    assert len(kq["chi_tiet"]) == 2


def test_tong_hop_khong_cong_don_khi_ca_2_deu_am():
    """Kiểm tra rõ ràng: KHÔNG được là tổng (-0.5 + -1.5 = -2.0), phải là MIN = -1.5."""
    sk1 = SuKien(nhom=NHOM_TAI_CHINH_TIEN_TE, muc_do="canh_bao_som")   # -0.5
    sk2 = SuKien(nhom=NHOM_THIEN_TAI, muc_do="cuc_bo")                 # -0.5
    kq = tinh_diem_su_kien_tong_hop([sk1, sk2])
    assert kq["score_event"] == -0.5  # MIN(-0.5, -0.5), chắc chắn KHÔNG phải -1.0
    assert kq["score_event"] != (-0.5 + -0.5)


def test_tong_hop_uu_tien_su_kien_tich_cuc_neu_khong_co_tieu_cuc():
    sk = SuKien(nhom=NHOM_DIA_CHINH_TRI, muc_do="giai_toa_hoan_toan")
    kq = tinh_diem_su_kien_tong_hop([sk])
    assert kq["score_event"] == 2.0


# ---------------------------------------------------------------------------
# Test cơ chế override Macro Score
# ---------------------------------------------------------------------------

def test_override_kich_hoat_khi_tieu_cuc_manh():
    kq = ap_dung_override_macro_score(macro_score_binh_thuong=0.8, score_event=-2.0)
    assert kq["da_ap_dung_override"] is True
    assert kq["macro_score_final"] == -1.0  # bị áp trần dù macro bình thường đang +0.8


def test_override_khong_kich_hoat_khi_su_kien_nhe():
    """Đúng case thực tế: score_event = -0.5 (cảnh báo sớm yên) KHÔNG đủ để
    kích hoạt override (-1.5), Macro Score giữ nguyên theo tính toán thường."""
    kq = ap_dung_override_macro_score(macro_score_binh_thuong=0.3, score_event=-0.5)
    assert kq["da_ap_dung_override"] is False
    assert kq["macro_score_final"] == 0.3


def test_override_bien_dung_nguong():
    kq_vua_du = ap_dung_override_macro_score(macro_score_binh_thuong=0.5, score_event=-1.5)
    assert kq_vua_du["da_ap_dung_override"] is True  # <= -1.5 tính là kích hoạt

    kq_chua_du = ap_dung_override_macro_score(macro_score_binh_thuong=0.5, score_event=-1.49)
    assert kq_chua_du["da_ap_dung_override"] is False


def test_override_khong_lam_xau_hon_neu_macro_da_thap_hon_tran():
    """Nếu macro_score_binh_thuong đã thấp hơn -1.0 sẵn, override không được
    kéo nó LÊN lại -1.0 (chỉ áp TRẦN, không phải áp SÀN)."""
    kq = ap_dung_override_macro_score(macro_score_binh_thuong=-1.8, score_event=-2.0)
    assert kq["macro_score_final"] == -1.8  # giữ nguyên mức xấu hơn, không kéo lên -1.0


# ---------------------------------------------------------------------------
# Test hàm tiện ích
# ---------------------------------------------------------------------------

def test_dung_su_kien_tu_dict():
    d = {
        "nhom": NHOM_TAI_CHINH_TIEN_TE,
        "muc_do": "canh_bao_som",
        "ghi_chu": "test",
        "ngay_ghi_nhan": "2026-08-02",
    }
    sk = dung_su_kien_tu_dict(d)
    assert sk.nhom == NHOM_TAI_CHINH_TIEN_TE
    assert sk.ghi_chu == "test"


def test_dung_su_kien_tu_dict_thieu_khoa_tuy_chon():
    d = {"nhom": NHOM_DIA_CHINH_TRI, "muc_do": "khong_co_su_kien"}
    sk = dung_su_kien_tu_dict(d)  # không lỗi dù thiếu ghi_chu/ngay_ghi_nhan
    assert sk.ghi_chu is None


def test_danh_sach_nhan_muc_do_hop_le():
    nhan = danh_sach_nhan_muc_do_hop_le(NHOM_TAI_CHINH_TIEN_TE)
    assert "canh_bao_som" in nhan
    assert "khung_hoang_toan_dien" in nhan
    assert len(nhan) == 6


def test_danh_sach_nhan_muc_do_nhom_khong_hop_le():
    with pytest.raises(NhomSuKienKhongHopLeError):
        danh_sach_nhan_muc_do_hop_le("nhom_sai")


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))
