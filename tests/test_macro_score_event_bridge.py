"""
tests/test_macro_score_event_bridge.py

Unit test cho core/macro_score_event_bridge.py — kiểm tra:
  1. Tương thích ngược: calculate_macro_score() GỐC (macro_score_engine.py)
     vẫn hoạt động bình thường, không bị ảnh hưởng bởi bridge.
  2. calculate_macro_score_v2() cho kết quả ĐÚNG khi có nhiều sự kiện đồng
     thời (case thực tế: Mỹ-Iran + cảnh báo rủi ro yên Nhật, 2/8/2026).
  3. Khi chỉ có 1 sự kiện, v2 phải cho macro_score TƯƠNG ĐƯƠNG bản gốc
     (đây là bài test quan trọng nhất để đảm bảo bridge không làm lệch kết
     quả so với hệ thống cũ khi không có tình huống đa sự kiện).
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.macro_score_engine import calculate_macro_score, InvalidMacroScoreError
from core.macro_score_event_bridge import (
    calculate_macro_score_v2,
    calculate_score_event_v2,
    su_kien_dia_chinh_tri_tu_nhan_cu,
    ANH_SANG_MUC_DO_NHOM_A,
)
from core.event_risk_classifier import (
    SuKien,
    NHOM_DIA_CHINH_TRI,
    NHOM_TAI_CHINH_TIEN_TE,
)


DU_LIEU_VI_MO_MAU = {
    "fed_rate_delta_last_meeting": 0.0,
    "fed_dotplot_delta": 0.1,
    "cpi_us_yoy": 3.5,
    "cpi_us_mom_3thang": [0.5, 0.6, -0.4],
    "cpi_vn_yoy": 4.69,
    "fx_ytd_change_pct": 1.2,
    "fx_so_tuan_tang_lien_tiep": 5,
    "fx_khoang_cach_dinh_pct": 0.3,
    "interbank_do_doc_duong_cong": 5.5,
    "interbank_thay_doi_tuan_3m": 0.2,
}


# ---------------------------------------------------------------------------
# 1. Tương thích ngược — bản GỐC vẫn chạy bình thường
# ---------------------------------------------------------------------------

def test_ban_goc_van_hoat_dong_binh_thuong():
    """calculate_macro_score() GỐC không hề bị ảnh hưởng bởi việc bridge tồn tại."""
    du_lieu = {**DU_LIEU_VI_MO_MAU, "su_kien_hien_tai": "escalating_tension"}
    kq = calculate_macro_score(du_lieu)
    assert "macro_score" in kq
    assert kq["chi_tiet_sub_scores"]["event"] == -1.0


# ---------------------------------------------------------------------------
# 2. calculate_macro_score_v2 với 1 sự kiện phải TƯƠNG ĐƯƠNG bản gốc
# ---------------------------------------------------------------------------

def test_v2_voi_1_su_kien_tuong_duong_ban_goc():
    """
    Đây là bài test quan trọng nhất: khi chỉ có 1 sự kiện (không có tình
    huống đa sự kiện), v2 phải cho CÙNG macro_score với bản gốc — chứng
    minh bridge không làm lệch kết quả so với hệ thống cũ.
    """
    su_kien_cu = "escalating_tension"
    du_lieu_goc = {**DU_LIEU_VI_MO_MAU, "su_kien_hien_tai": su_kien_cu}
    kq_goc = calculate_macro_score(du_lieu_goc)

    su_kien_moi = su_kien_dia_chinh_tri_tu_nhan_cu(su_kien_cu)
    kq_v2 = calculate_macro_score_v2(DU_LIEU_VI_MO_MAU, cac_su_kien=[su_kien_moi])

    assert kq_v2["macro_score"] == kq_goc["macro_score"]
    assert kq_v2["nhan"] == kq_goc["nhan"]
    assert kq_v2["chi_tiet_sub_scores"]["event"] == kq_goc["chi_tiet_sub_scores"]["event"]


@pytest.mark.parametrize("nhan_cu", list(ANH_SANG_MUC_DO_NHOM_A.keys()))
def test_v2_tuong_duong_ban_goc_cho_tat_ca_nhan_cu(nhan_cu):
    """Lặp lại test trên cho TẤT CẢ 5 nhãn cũ, đảm bảo ánh xạ đúng toàn bộ bảng."""
    du_lieu_goc = {**DU_LIEU_VI_MO_MAU, "su_kien_hien_tai": nhan_cu}
    kq_goc = calculate_macro_score(du_lieu_goc)

    su_kien_moi = su_kien_dia_chinh_tri_tu_nhan_cu(nhan_cu)
    kq_v2 = calculate_macro_score_v2(DU_LIEU_VI_MO_MAU, cac_su_kien=[su_kien_moi])

    assert kq_v2["macro_score"] == kq_goc["macro_score"]
    assert kq_v2["nhan"] == kq_goc["nhan"]


def test_anh_xa_nhan_sai_raise_error():
    with pytest.raises(InvalidMacroScoreError):
        su_kien_dia_chinh_tri_tu_nhan_cu("nhan_khong_ton_tai")


# ---------------------------------------------------------------------------
# 3. Case thực tế: Mỹ-Iran + cảnh báo rủi ro yên Nhật ĐỒNG THỜI (2/8/2026)
# ---------------------------------------------------------------------------

def test_case_thuc_te_my_iran_va_canh_bao_yen_dong_thoi():
    """
    Xung đột Mỹ-Iran (Nhóm A, 'cang_thang_leo_thang' = -1.0) đang diễn ra
    ĐỒNG THỜI với cảnh báo sớm rủi ro can thiệp yên Nhật (Nhóm B,
    'canh_bao_som' = -0.5).
    -> score_event phải = MIN(-1.0, -0.5) = -1.0 (không phải -1.5 nếu cộng dồn).
    -> Với score_event = -1.0 (> -1.5), KHÔNG kích hoạt override.
    """
    su_kien_iran = SuKien(nhom=NHOM_DIA_CHINH_TRI, muc_do="cang_thang_leo_thang")
    su_kien_yen = SuKien(nhom=NHOM_TAI_CHINH_TIEN_TE, muc_do="canh_bao_som")

    kq = calculate_macro_score_v2(
        DU_LIEU_VI_MO_MAU, cac_su_kien=[su_kien_iran, su_kien_yen]
    )

    assert kq["chi_tiet_sub_scores"]["event"] == -1.0
    assert len(kq["chi_tiet_su_kien"]) == 2
    assert kq["su_kien_quyet_dinh"]["nhom"] == NHOM_DIA_CHINH_TRI


def test_case_xung_dot_no_ra_kich_hoat_override_du_co_them_su_kien_nhe():
    """
    Nếu Mỹ-Iran leo thang thành 'xung_dot_no_ra' (-2.0) trong khi vẫn có
    cảnh báo yên nhẹ (-0.5) -> score_event = MIN(-2.0, -0.5) = -2.0
    -> phải kích hoạt override (macro_score bị áp trần -1.0).
    """
    su_kien_iran = SuKien(nhom=NHOM_DIA_CHINH_TRI, muc_do="xung_dot_no_ra")
    su_kien_yen = SuKien(nhom=NHOM_TAI_CHINH_TIEN_TE, muc_do="canh_bao_som")

    # Dữ liệu vĩ mô giả định các chỉ số khác đang RẤT TỐT (+2 cả) để kiểm tra
    # override có thực sự ép macro_score xuống -1.0 hay không, bất kể tốt xấu.
    du_lieu_rat_tot = {
        "fed_rate_delta_last_meeting": -0.5, "fed_dotplot_delta": -0.5,
        "cpi_us_yoy": 1.0, "cpi_us_mom_3thang": [-0.5, -0.5, -0.5],
        "cpi_vn_yoy": 2.0,
        "fx_ytd_change_pct": -3.0, "fx_so_tuan_tang_lien_tiep": 0, "fx_khoang_cach_dinh_pct": 5.0,
        "interbank_do_doc_duong_cong": -3.0, "interbank_thay_doi_tuan_3m": -0.5,
    }

    kq = calculate_macro_score_v2(
        du_lieu_rat_tot, cac_su_kien=[su_kien_iran, su_kien_yen]
    )

    assert kq["chi_tiet_sub_scores"]["event"] == -2.0
    assert kq["macro_score"] == -1.0  # bị áp trần dù các chỉ số khác đều tốt


def test_khong_co_su_kien_nao_tuong_duong_none():
    kq_v2 = calculate_macro_score_v2(DU_LIEU_VI_MO_MAU, cac_su_kien=[])
    du_lieu_goc = {**DU_LIEU_VI_MO_MAU, "su_kien_hien_tai": "none"}
    kq_goc = calculate_macro_score(du_lieu_goc)

    assert kq_v2["macro_score"] == kq_goc["macro_score"]
    assert kq_v2["chi_tiet_sub_scores"]["event"] == 0.0


def test_calculate_score_event_v2_truc_tiep():
    su_kien = SuKien(nhom=NHOM_TAI_CHINH_TIEN_TE, muc_do="dau_hieu_unwind")
    kq = calculate_score_event_v2([su_kien])
    assert kq["score_event"] == -1.5


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))
