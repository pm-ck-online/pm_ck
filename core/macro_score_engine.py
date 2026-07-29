"""
macro_score_engine.py
========================
[Bổ sung — Module Tính Điểm Vĩ Mô (Macro Score Engine)]

Lớp 1 của mô hình xác định trạng thái thị trường — tính MỘT điểm số tổng
hợp duy nhất (Macro Score, khoảng [-2, +2]) phản ánh mức độ thuận
lợi/bất lợi của bối cảnh vĩ mô, dựa trên 6 nhóm chỉ số:
    1. Fed Funds Rate + dot-plot
    2. CPI Mỹ
    3. CPI Việt Nam
    4. Tỷ giá USD/VND
    5. Lãi suất liên ngân hàng VN
    6. Sự kiện địa chính trị / rủi ro đột biến (có cơ chế OVERRIDE)

Macro Score dùng làm "trần giới hạn" cho Module Xác định Trạng thái Thị
trường: điểm càng thấp thì càng giới hạn khả năng gán nhãn UPTREND, bất
kể tín hiệu kỹ thuật ở Lớp 2/3 tích cực ra sao.

LƯU Ý QUAN TRỌNG: các hằng số chuẩn hóa (2.0%, 0.3%, 3.0%, 10 tuần...) và
trọng số mặc định trong module này là ĐIỂM KHỞI ĐẦU THAM KHẢO theo tài
liệu kỹ thuật gốc — BẮT BUỘC backtest lại trên dữ liệu lịch sử 3-5 năm
trước khi dùng cho quyết định thực tế (xem mục 7 tài liệu gốc).

KHÔNG dùng Macro Score như tín hiệu giao dịch độc lập — chỉ dùng làm lớp
giới hạn/điều chỉnh cho module xác định giai đoạn thị trường.
"""

from __future__ import annotations

from typing import Optional


class InvalidMacroScoreError(ValueError):
    """Dữ liệu đầu vào không hợp lệ cho module tính điểm vĩ mô."""


def clip(x: float, lo: float = -2.0, hi: float = 2.0) -> float:
    """Giới hạn giá trị trong khoảng [lo, hi] — tránh 1 điểm dữ liệu bất
    thường làm méo điểm số (mục 2.2 tài liệu).
    """
    return max(lo, min(hi, x))


# ==============================================================================
# TRỌNG SỐ MẶC ĐỊNH (mục 3 tài liệu — cần backtest hiệu chỉnh lại)
# ==============================================================================

DEFAULT_WEIGHTS = {
    "fed": 0.25,
    "cpi_us": 0.15,
    "cpi_vn": 0.15,
    "fx": 0.20,
    "interbank": 0.10,
    "event": 0.15,
}

# --- Bảng phân loại mức độ sự kiện địa chính trị (mục 2.6) ---
EVENT_SCORE_TABLE = {
    "none": 0.0,                     # Không có sự kiện rủi ro nổi bật
    "escalating_tension": -1.0,      # Căng thẳng leo thang giai đoạn đầu
    "conflict_outbreak": -2.0,       # Xung đột/chiến sự nổ ra hoặc leo thang mạnh
    "de_escalation_signal": 1.0,     # Có tín hiệu hạ nhiệt/đàm phán tiến triển
    "positive_resolution": 2.0,      # Sự kiện tích cực xác nhận, rủi ro giải tỏa
}


# ==============================================================================
# HÀM ĐIỂM CON (sub-score) CHO TỪNG CHỈ SỐ — mục 2 tài liệu
# ==============================================================================

def f_direction(delta_rate_last: float) -> float:
    """Điểm theo hướng thay đổi Fed Funds Rate tại cuộc họp FOMC gần nhất
    (mục 2.1): cắt giảm -> +2, giữ nguyên -> 0, tăng -> -2.
    """
    if delta_rate_last < 0:
        return 2.0
    if delta_rate_last > 0:
        return -2.0
    return 0.0


def f_dotplot(delta_dot_plot: float) -> float:
    """Điểm theo độ lệch dot-plot kỳ vọng (mục 2.1): dịch chuyển ôn hòa
    hơn -> +2, không đổi -> 0, dịch chuyển diều hâu hơn -> -2.
    """
    if delta_dot_plot < 0:
        return 2.0
    if delta_dot_plot > 0:
        return -2.0
    return 0.0


def calculate_score_fed(
    delta_rate_last: float, delta_dot_plot: float, a1: float = 0.4, a2: float = 0.6
) -> float:
    """score_fed = a1 * f_direction + a2 * f_dotplot (mục 2.1).

    Trọng số mặc định a2 > a1 vì dot-plot phản ánh KỲ VỌNG tương lai, có
    ảnh hưởng dài hạn hơn quyết định tức thời.
    """
    return a1 * f_direction(delta_rate_last) + a2 * f_dotplot(delta_dot_plot)


def calculate_score_cpi_us(
    cpi_yoy: float, cpi_mom_3thang: list[float], b1: float = 0.5, b2: float = 0.5
) -> float:
    """score_cpi_us dựa trên khoảng cách tới mục tiêu 2% + momentum 3
    tháng gần nhất (mục 2.2).
    """
    if not cpi_mom_3thang:
        raise InvalidMacroScoreError("cpi_mom_3thang không được rỗng.")

    khoang_cach_muc_tieu = cpi_yoy - 2.0
    momentum_3m = sum(cpi_mom_3thang) / len(cpi_mom_3thang)

    return (
        -b1 * clip(khoang_cach_muc_tieu / 2.0)
        - b2 * clip(momentum_3m / 0.3)
    )


def calculate_score_cpi_vn(
    cpi_yoy_vn: float, muc_tieu_cpi_vn: float = 4.0, c1: float = 1.0
) -> float:
    """score_cpi_vn dựa trên khoảng cách CPI thực tế so với mục tiêu kiểm
    soát lạm phát (mục 2.3). `muc_tieu_cpi_vn` KHÔNG hardcode vĩnh viễn —
    cần cập nhật theo Nghị quyết Quốc hội từng năm.
    """
    khoang_cach = cpi_yoy_vn - muc_tieu_cpi_vn
    return -c1 * clip(khoang_cach / 2.0)


def calculate_score_fx(
    fx_ytd_change_pct: float,
    fx_so_tuan_tang_lien_tiep: float,
    fx_khoang_cach_dinh_pct: float,
    d1: float = 0.4, d2: float = 0.3, d3: float = 0.3,
) -> float:
    """score_fx dựa trên 3 thành phần: biến động YTD, số tuần tăng liên
    tiếp, và khoảng cách tới đỉnh lịch sử (mục 2.4).
    """
    term_ytd = clip(fx_ytd_change_pct / 3.0)
    term_consecutive_weeks = clip(fx_so_tuan_tang_lien_tiep / 10, 0.0, 2.0)
    term_near_peak = clip((2.0 - fx_khoang_cach_dinh_pct) / 2.0)

    return -d1 * term_ytd - d2 * term_consecutive_weeks - d3 * term_near_peak


def calculate_score_interbank(
    interbank_do_doc_duong_cong: float,
    interbank_thay_doi_tuan_3m: float,
    e1: float = 0.5, e2: float = 0.5,
) -> float:
    """score_interbank dựa trên độ dốc đường cong lãi suất (spread 3
    tháng - qua đêm) và thay đổi tuần của lãi suất 3 tháng (mục 2.5).
    """
    return (
        -e1 * clip(interbank_do_doc_duong_cong / 3.0)
        - e2 * clip(interbank_thay_doi_tuan_3m / 0.5)
    )


def calculate_score_event(su_kien_hien_tai: str) -> float:
    """Tra bảng phân loại mức độ sự kiện địa chính trị (mục 2.6).

    Đây là điểm DUY NHẤT có thể override toàn bộ Macro Score về mức rất
    âm bất kể các chỉ số khác — vì độ trễ phản ánh vào dữ liệu kinh tế
    luôn chậm hơn phản ứng thị trường thực tế.
    """
    if su_kien_hien_tai not in EVENT_SCORE_TABLE:
        raise InvalidMacroScoreError(
            f"su_kien_hien_tai '{su_kien_hien_tai}' không hợp lệ. Cần một "
            f"trong {sorted(EVENT_SCORE_TABLE.keys())}."
        )
    return EVENT_SCORE_TABLE[su_kien_hien_tai]


# ==============================================================================
# PHÂN LOẠI NHÃN CUỐI CÙNG (mục 4 tài liệu)
# ==============================================================================

def classify_macro_score(macro_score: float, score_event: float) -> str:
    """Phân loại nhãn TÍCH_CUC/TRUNG_TINH/TIEU_CUC/TIEU_CUC_MANH theo
    ĐÚNG THỨ TỰ kiểm tra trong tài liệu (mục 4 + pseudo-code mục 5).

    LƯU Ý: `macro_score` truyền vào đây PHẢI là giá trị ĐÃ ÁP DỤNG cơ chế
    override sự kiện (xem `calculate_macro_score()`), không phải điểm
    thô trước khi override.
    """
    if macro_score >= 0.5:
        return "TICH_CUC"
    if macro_score < -1.0 and score_event <= -1.5:
        return "TIEU_CUC_MANH"
    if macro_score < -0.5:
        return "TIEU_CUC"
    return "TRUNG_TINH"


# ==============================================================================
# HÀM CHÍNH — tổng hợp Macro Score
# ==============================================================================

REQUIRED_FIELDS = {
    "fed_rate_delta_last_meeting", "fed_dotplot_delta",
    "cpi_us_yoy", "cpi_us_mom_3thang",
    "cpi_vn_yoy",
    "fx_ytd_change_pct", "fx_so_tuan_tang_lien_tiep", "fx_khoang_cach_dinh_pct",
    "interbank_do_doc_duong_cong", "interbank_thay_doi_tuan_3m",
    "su_kien_hien_tai",
}


def calculate_macro_score(du_lieu_vi_mo: dict, weights: Optional[dict] = None) -> dict:
    """Tính Macro Score tổng hợp từ 6 nhóm chỉ số (mục 3 + mục 5 pseudo-code).

    `du_lieu_vi_mo` cần đủ các trường trong `REQUIRED_FIELDS` (xem mục 5
    tài liệu gốc để biết ý nghĩa từng trường). `muc_tieu_cpi_vn` là tùy
    chọn (mặc định 4.0%, theo Nghị quyết Quốc hội — nên truyền giá trị
    cập nhật hàng năm thay vì dùng mặc định).

    Trả về:
        {"macro_score": float, "nhan": str,
         "chi_tiet_sub_scores": {"fed", "cpi_us", "cpi_vn", "fx",
                                  "interbank", "event"}}
    """
    missing = REQUIRED_FIELDS - set(du_lieu_vi_mo.keys())
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
    score_event = calculate_score_event(du_lieu_vi_mo["su_kien_hien_tai"])

    macro_score = (
        weights["fed"] * score_fed
        + weights["cpi_us"] * score_cpi_us
        + weights["cpi_vn"] * score_cpi_vn
        + weights["fx"] * score_fx
        + weights["interbank"] * score_interbank
        + weights["event"] * score_event
    )

    # --- Cơ chế OVERRIDE khi có rủi ro sự kiện nghiêm trọng (mục 2.6) ---
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
    }
