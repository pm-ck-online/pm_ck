"""
core/macro_score_event_bridge.py

Module CẦU NỐI (bridge) — tích hợp `event_risk_classifier.py` (phân loại
sự kiện ĐA NHÓM: địa chính trị, tài chính-tiền tệ, thiên tai) vào
`macro_score_engine.py` THẬT đã có sẵn trong dự án, MÀ KHÔNG SỬA file gốc.

Lý do thiết kế "không xâm lấn": `macro_score_engine.py` hiện tại dùng
`EVENT_SCORE_TABLE` với key TIẾNG ANH ("none", "escalating_tension",
"conflict_outbreak", "de_escalation_signal", "positive_resolution") và
hàm `calculate_score_event(su_kien_hien_tai: str)` chỉ nhận 1 sự kiện đơn
lẻ. Bất kỳ code/test nào trong dự án đang gọi trực tiếp
`calculate_macro_score()`/`calculate_score_event()` với 1 string sẽ TIẾP
TỤC HOẠT ĐỘNG BÌNH THƯỜNG — bridge này chỉ THÊM khả năng mới (đa nhóm sự
kiện), không thay thế.

Cách dùng:
  - Nếu chỉ có 1 sự kiện kiểu cũ (string) -> gọi calculate_macro_score() gốc
    như bình thường, KHÔNG cần dùng module này.
  - Nếu có NHIỀU sự kiện đồng thời (khác nhóm, vd: xung đột Mỹ-Iran + cảnh
    báo rủi ro yên Nhật cùng lúc) -> dùng `calculate_macro_score_v2()` ở
    đây, truyền vào `cac_su_kien` (list[SuKien]) thay vì `su_kien_hien_tai`.
"""

from __future__ import annotations

from typing import Optional

from core.macro_score_engine import (
    DEFAULT_WEIGHTS,
    InvalidMacroScoreError,
    calculate_score_fed,
    calculate_score_cpi_us,
    calculate_score_cpi_vn,
    calculate_score_fx,
    calculate_score_interbank,
    classify_macro_score,
)
from core.event_risk_classifier import (
    SuKien,
    tinh_diem_su_kien_tong_hop,
    NHOM_DIA_CHINH_TRI,
)


# ==============================================================================
# 1. Ánh xạ nhãn CŨ (tiếng Anh, trong EVENT_SCORE_TABLE của macro_score_engine.py
#    gốc) sang nhãn MỚI (tiếng Việt, trong BANG_DIA_CHINH_TRI của
#    event_risk_classifier.py) — để đảm bảo dữ liệu/cấu hình cũ vẫn map đúng
#    khi ai đó vẫn quen dùng nhãn tiếng Anh.
# ==============================================================================

ANH_SANG_MUC_DO_NHOM_A = {
    "none": "khong_co_su_kien",
    "escalating_tension": "cang_thang_leo_thang",
    "conflict_outbreak": "xung_dot_no_ra",
    "de_escalation_signal": "ha_nhiet_dam_phan",
    "positive_resolution": "giai_toa_hoan_toan",
}


def su_kien_dia_chinh_tri_tu_nhan_cu(nhan_cu: str, **kwargs) -> SuKien:
    """
    Dựng 1 SuKien Nhóm A (địa chính trị) từ nhãn TIẾNG ANH cũ — dùng khi
    muốn tái sử dụng dữ liệu cấu hình/log cũ đang lưu theo EVENT_SCORE_TABLE
    gốc mà không cần sửa lại toàn bộ dữ liệu đã có.
    """
    if nhan_cu not in ANH_SANG_MUC_DO_NHOM_A:
        raise InvalidMacroScoreError(
            f"Nhãn cũ '{nhan_cu}' không nằm trong ánh xạ. Cần một trong "
            f"{sorted(ANH_SANG_MUC_DO_NHOM_A.keys())}."
        )
    return SuKien(nhom=NHOM_DIA_CHINH_TRI, muc_do=ANH_SANG_MUC_DO_NHOM_A[nhan_cu], **kwargs)


# ==============================================================================
# 2. Tính score_event ĐA NHÓM (thay thế calculate_score_event khi cần)
# ==============================================================================

def calculate_score_event_v2(cac_su_kien: list[SuKien]) -> dict:
    """
    Phiên bản đa nhóm của `calculate_score_event()` gốc — nhận DANH SÁCH
    sự kiện (có thể khác nhóm, xảy ra đồng thời), trả về score_event tổng
    hợp = MIN (mức xấu nhất), kèm chi tiết từng sự kiện để minh bạch.

    Nếu chỉ có 1 sự kiện, kết quả tương đương gọi calculate_score_event()
    gốc cho đúng sự kiện đó (không có gì khác biệt về giá trị điểm).
    """
    return tinh_diem_su_kien_tong_hop(cac_su_kien)


# ==============================================================================
# 3. calculate_macro_score PHIÊN BẢN 2 — dùng đa nhóm sự kiện
# ==============================================================================

REQUIRED_FIELDS_V2 = {
    "fed_rate_delta_last_meeting", "fed_dotplot_delta",
    "cpi_us_yoy", "cpi_us_mom_3thang",
    "cpi_vn_yoy",
    "fx_ytd_change_pct", "fx_so_tuan_tang_lien_tiep", "fx_khoang_cach_dinh_pct",
    "interbank_do_doc_duong_cong", "interbank_thay_doi_tuan_3m",
    # LƯU Ý: KHÔNG cần "su_kien_hien_tai" nữa — đã thay bằng tham số
    # `cac_su_kien` riêng, xem hàm bên dưới.
}


def calculate_macro_score_v2(
    du_lieu_vi_mo: dict,
    cac_su_kien: Optional[list[SuKien]] = None,
    weights: Optional[dict] = None,
) -> dict:
    """
    Bản sao của `calculate_macro_score()` gốc trong macro_score_engine.py,
    CHỈ khác ở cách tính `score_event`: dùng cơ chế đa nhóm (Mục 2.6 đã
    cập nhật) thay vì tra 1 bảng đơn nhóm.

    5 nhóm chỉ số còn lại (fed, cpi_us, cpi_vn, fx, interbank) TÁI SỬ DỤNG
    NGUYÊN VẸN các hàm THẬT từ core/macro_score_engine.py (import ở đầu
    file) — không viết lại logic, tránh sai lệch/trùng lặp.

    `cac_su_kien`: list[SuKien] — nếu None hoặc rỗng, coi như KHÔNG có sự
    kiện rủi ro nào (score_event = 0.0), tương đương "none" ở bảng cũ.

    Trả về CÙNG CẤU TRÚC với `calculate_macro_score()` gốc, cộng thêm khóa
    "chi_tiet_su_kien" (danh sách chi tiết từng sự kiện đã xét) để minh bạch.
    """
    missing = REQUIRED_FIELDS_V2 - set(du_lieu_vi_mo.keys())
    if missing:
        raise InvalidMacroScoreError(
            f"du_lieu_vi_mo thiếu các trường bắt buộc: {sorted(missing)}."
        )

    weights = weights or DEFAULT_WEIGHTS

    score_fed = calculate_score_fed(
        du_lieu_vi_mo["fed_rate_delta_last_meeting"],
        du_lieu_vi_mo["fed_dotplot_delta"],
    )
    score_cpi_us = calculate_score_cpi_us(
        du_lieu_vi_mo["cpi_us_yoy"], du_lieu_vi_mo["cpi_us_mom_3thang"],
    )
    score_cpi_vn = calculate_score_cpi_vn(
        du_lieu_vi_mo["cpi_vn_yoy"], du_lieu_vi_mo.get("muc_tieu_cpi_vn", 4.0),
    )
    score_fx = calculate_score_fx(
        du_lieu_vi_mo["fx_ytd_change_pct"],
        du_lieu_vi_mo["fx_so_tuan_tang_lien_tiep"],
        du_lieu_vi_mo["fx_khoang_cach_dinh_pct"],
    )
    score_interbank = calculate_score_interbank(
        du_lieu_vi_mo["interbank_do_doc_duong_cong"],
        du_lieu_vi_mo["interbank_thay_doi_tuan_3m"],
    )

    ket_qua_su_kien = calculate_score_event_v2(cac_su_kien or [])
    score_event = ket_qua_su_kien["score_event"]

    macro_score = (
        weights["fed"] * score_fed
        + weights["cpi_us"] * score_cpi_us
        + weights["cpi_vn"] * score_cpi_vn
        + weights["fx"] * score_fx
        + weights["interbank"] * score_interbank
        + weights["event"] * score_event
    )

    # --- Cơ chế OVERRIDE khi có rủi ro sự kiện nghiêm trọng (giữ nguyên
    #     ngưỡng -1.5 / trần -1.0 như bản gốc, chỉ đổi nguồn score_event) ---
    if score_event <= -1.5:
        macro_score = min(macro_score, -1.0)

    nhan = classify_macro_score(macro_score, score_event)

    return {
        "macro_score": round(macro_score, 3),
        "nhan": nhan,
        "chi_tiet_sub_scores": {
            "fed": round(score_fed, 3),
            "cpi_us": round(score_cpi_us, 3),
            "cpi_vn": round(score_cpi_vn, 3),
            "fx": round(score_fx, 3),
            "interbank": round(score_interbank, 3),
            "event": round(score_event, 3),
        },
        "chi_tiet_su_kien": ket_qua_su_kien["chi_tiet"],
        "su_kien_quyet_dinh": ket_qua_su_kien["su_kien_quyet_dinh"],
    }
