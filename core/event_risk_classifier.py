"""
core/event_risk_classifier.py

Phân loại "Sự kiện rủi ro đột biến" (score_event) cho Macro Score Engine —
PHIÊN BẢN ĐA NHÓM (Mục 2.6 đã cập nhật trong prompt gốc), thay thế bảng đơn
nhóm ban đầu.

Bối cảnh cập nhật: bảng đơn nhóm ban đầu chỉ có "Địa chính trị/Quân sự",
không phù hợp với sự kiện thực tế phát sinh 2/8/2026 (Mỹ-Nhật phối hợp can
thiệp tỷ giá yên — một sự kiện RỦI RO TÀI CHÍNH-TIỀN TỆ, không phải xung đột
quân sự, và bản thân hành động can thiệp là ỔN ĐỊNH chứ không phải LEO THANG).

Thiết kế: nhiều NHÓM sự kiện độc lập (địa chính trị, tài chính-tiền tệ,
thiên tai/dịch bệnh...), mỗi nhóm có bảng mức độ riêng; khi có nhiều sự kiện
đồng thời (khác nhóm), lấy MIN (mức xấu nhất) làm score_event cuối cùng,
không cộng dồn.

Đây là công cụ HỖ TRỢ TÍNH TOÁN THAM KHẢO — không phải khuyến nghị đầu tư.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional


# ==============================================================================
# 1. Định nghĩa các NHÓM sự kiện
# ==============================================================================

NHOM_DIA_CHINH_TRI = "dia_chinh_tri"
NHOM_TAI_CHINH_TIEN_TE = "tai_chinh_tien_te"
NHOM_THIEN_TAI = "thien_tai"

CAC_NHOM_HOP_LE = (NHOM_DIA_CHINH_TRI, NHOM_TAI_CHINH_TIEN_TE, NHOM_THIEN_TAI)


class NhomSuKienKhongHopLeError(ValueError):
    """Nhóm sự kiện không nằm trong danh sách CAC_NHOM_HOP_LE."""


class MucDoKhongHopLeError(ValueError):
    """Mức độ không tồn tại trong bảng của nhóm sự kiện tương ứng."""


# ==============================================================================
# 2. Bảng phân loại mức độ — Nhóm A: Địa chính trị / Quân sự
# ==============================================================================

BANG_DIA_CHINH_TRI = {
    "khong_co_su_kien": {
        "diem": 0.0,
        "mo_ta": "Không có sự kiện rủi ro nổi bật — giai đoạn bình thường.",
    },
    "cang_thang_leo_thang": {
        "diem": -1.0,
        "mo_ta": "Căng thẳng leo thang (giai đoạn đầu) — đe dọa quân sự, đàm phán căng thẳng.",
    },
    "xung_dot_no_ra": {
        "diem": -2.0,
        "mo_ta": "Xung đột/chiến sự nổ ra hoặc leo thang mạnh (vd: Mỹ-Israel tấn công Iran, đóng eo biển Hormuz).",
    },
    "ha_nhiet_dam_phan": {
        "diem": 1.0,
        "mo_ta": "Có tín hiệu hạ nhiệt/đàm phán tiến triển (vd: tuyên bố xung đột 'sắp kết thúc').",
    },
    "giai_toa_hoan_toan": {
        "diem": 2.0,
        "mo_ta": "Sự kiện tích cực xác nhận, rủi ro giải tỏa (vd: ký kết ngừng bắn chính thức).",
    },
}

# ==============================================================================
# 3. Bảng phân loại mức độ — Nhóm B: Tài chính - Tiền tệ Toàn cầu (MỚI)
# ==============================================================================

BANG_TAI_CHINH_TIEN_TE = {
    "binh_thuong": {
        "diem": 0.0,
        "mo_ta": "Bình thường — không có rủi ro tài chính-tiền tệ nổi bật.",
    },
    "canh_bao_som": {
        "diem": -0.5,
        "mo_ta": (
            "Cảnh báo sớm / rủi ro tiềm ẩn được nhận diện, đang có biện pháp "
            "ngăn chặn (vd: Mỹ-Nhật phối hợp can thiệp cứu yên 2/8/2026 — rủi ro "
            "carry-trade unwind được nhận diện sớm, CHƯA xảy ra thực tế)."
        ),
    },
    "dau_hieu_unwind": {
        "diem": -1.5,
        "mo_ta": (
            "Bắt đầu có dấu hiệu rút vốn/unwind thực tế (vd: VIX tăng vọt, "
            "Nikkei giảm mạnh sau can thiệp, dòng vốn rút khỏi thị trường mới "
            "nổi tăng tốc)."
        ),
    },
    "khung_hoang_toan_dien": {
        "diem": -2.5,
        "mo_ta": (
            "Khủng hoảng toàn diện đã xảy ra (tiền lệ: Nikkei sập >10%/phiên "
            "8/2024, khủng hoảng ngân hàng khu vực SVB 2023)."
        ),
    },
    "can_thiep_thanh_cong": {
        "diem": 1.0,
        "mo_ta": "Can thiệp thành công, ổn định trở lại (tỷ giá/lợi suất trái phiếu ổn định sau vài phiên).",
    },
    "giai_toa_xac_nhan": {
        "diem": 1.5,
        "mo_ta": "Rủi ro giải tỏa hoàn toàn, xác nhận qua dữ liệu (không còn dấu hiệu unwind sau 2-4 tuần theo dõi).",
    },
}

# ==============================================================================
# 4. Bảng phân loại mức độ — Nhóm C: Thiên tai / Dịch bệnh (dự phòng, khung tối thiểu)
# ==============================================================================

BANG_THIEN_TAI = {
    "binh_thuong": {
        "diem": 0.0,
        "mo_ta": "Bình thường.",
    },
    "cuc_bo": {
        "diem": -0.5,
        "mo_ta": "Sự kiện cục bộ, ảnh hưởng hạn chế (thiên tai/dịch bệnh quy mô khu vực).",
    },
    "dien_rong": {
        "diem": -2.0,
        "mo_ta": (
            "Sự kiện diện rộng, gián đoạn chuỗi cung ứng/kinh tế rõ rệt "
            "(tiền lệ: COVID-19 giai đoạn đầu 2020)."
        ),
    },
}
# Ghi chú: Nhóm C chỉ là khung tối thiểu dự phòng — cần mở rộng chi tiết hơn
# nếu có sự kiện thực tế phát sinh, theo đúng nguyên tắc "cập nhật khi có
# bằng chứng thực tế" đã áp dụng để tạo ra Nhóm B.

_BANG_THEO_NHOM = {
    NHOM_DIA_CHINH_TRI: BANG_DIA_CHINH_TRI,
    NHOM_TAI_CHINH_TIEN_TE: BANG_TAI_CHINH_TIEN_TE,
    NHOM_THIEN_TAI: BANG_THIEN_TAI,
}


# ==============================================================================
# 5. Cấu trúc dữ liệu 1 sự kiện + tra bảng điểm
# ==============================================================================

@dataclass(frozen=True)
class SuKien:
    """Một sự kiện rủi ro cụ thể đang diễn ra, cần được chấm điểm."""

    nhom: str
    muc_do: str
    ghi_chu: Optional[str] = None
    ngay_ghi_nhan: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.nhom not in CAC_NHOM_HOP_LE:
            raise NhomSuKienKhongHopLeError(
                f"Nhóm '{self.nhom}' không hợp lệ. Các nhóm hợp lệ: {CAC_NHOM_HOP_LE}"
            )
        bang = _BANG_THEO_NHOM[self.nhom]
        if self.muc_do not in bang:
            raise MucDoKhongHopLeError(
                f"Mức độ '{self.muc_do}' không tồn tại trong bảng nhóm '{self.nhom}'. "
                f"Các mức hợp lệ: {list(bang.keys())}"
            )


def tra_diem_su_kien(su_kien: SuKien) -> float:
    """Tra điểm của 1 sự kiện theo đúng bảng của nhóm tương ứng."""
    bang = _BANG_THEO_NHOM[su_kien.nhom]
    return bang[su_kien.muc_do]["diem"]


def tra_mo_ta_su_kien(su_kien: SuKien) -> str:
    bang = _BANG_THEO_NHOM[su_kien.nhom]
    return bang[su_kien.muc_do]["mo_ta"]


# ==============================================================================
# 6. Tổng hợp NHIỀU sự kiện đồng thời (khác nhóm) — lấy MIN (mức xấu nhất)
# ==============================================================================

def tinh_diem_su_kien_tong_hop(cac_su_kien: list[SuKien]) -> dict:
    """
    Nhận danh sách các sự kiện ĐANG DIỄN RA ĐỒNG THỜI (có thể khác nhóm),
    trả về:
      - score_event: điểm tổng hợp cuối cùng = MIN(điểm từng sự kiện)
        (LẤY MỨC XẤU NHẤT, KHÔNG CỘNG DỒN — xem lý do ở Mục 2.6 prompt gốc)
      - su_kien_quyet_dinh: sự kiện nào đang "quyết định" điểm cuối (điểm thấp nhất)
      - chi_tiet: điểm + mô tả của TỪNG sự kiện, để minh bạch cho người dùng

    Nếu danh sách rỗng -> coi như KHÔNG có sự kiện rủi ro (score_event = 0.0).
    """
    if not cac_su_kien:
        return {
            "score_event": 0.0,
            "su_kien_quyet_dinh": None,
            "chi_tiet": [],
        }

    chi_tiet = []
    diem_min = None
    su_kien_min = None

    for sk in cac_su_kien:
        diem = tra_diem_su_kien(sk)
        mo_ta = tra_mo_ta_su_kien(sk)
        chi_tiet.append(
            {
                "nhom": sk.nhom,
                "muc_do": sk.muc_do,
                "diem": diem,
                "mo_ta": mo_ta,
                "ghi_chu": sk.ghi_chu,
                "ngay_ghi_nhan": sk.ngay_ghi_nhan,
            }
        )
        if diem_min is None or diem < diem_min:
            diem_min = diem
            su_kien_min = sk

    return {
        "score_event": diem_min,
        "su_kien_quyet_dinh": (
            {"nhom": su_kien_min.nhom, "muc_do": su_kien_min.muc_do, "diem": diem_min}
            if su_kien_min is not None
            else None
        ),
        "chi_tiet": chi_tiet,
    }


# ==============================================================================
# 7. Cơ chế override toàn bộ Macro Score (đúng Mục 2.6 prompt gốc)
# ==============================================================================

NGUONG_OVERRIDE_TIEU_CUC_MANH = -1.5
TRAN_MACRO_SCORE_KHI_OVERRIDE = -1.0


def ap_dung_override_macro_score(macro_score_binh_thuong: float, score_event: float) -> dict:
    """
    Áp dụng cơ chế override: nếu score_event <= -1.5 (tiêu cực mạnh), Macro
    Score cuối cùng bị áp TRẦN ở -1.0 bất kể các chỉ số khác tính ra sao —
    vì độ trễ phản ánh vào dữ liệu kinh tế (CPI, lãi suất...) luôn chậm hơn
    phản ứng thị trường tài chính thực tế.
    """
    da_override = score_event <= NGUONG_OVERRIDE_TIEU_CUC_MANH
    macro_score_final = (
        min(macro_score_binh_thuong, TRAN_MACRO_SCORE_KHI_OVERRIDE)
        if da_override
        else macro_score_binh_thuong
    )
    return {
        "macro_score_final": round(macro_score_final, 3),
        "da_ap_dung_override": da_override,
        "nguong_override": NGUONG_OVERRIDE_TIEU_CUC_MANH,
    }


# ==============================================================================
# 8. Hàm tiện ích: dựng nhanh SuKien từ dict thô (phục vụ nạp dữ liệu ngoài/JSON)
# ==============================================================================

def dung_su_kien_tu_dict(d: dict) -> SuKien:
    """
    d cần có tối thiểu 2 khóa: "nhom", "muc_do". Có thể có thêm "ghi_chu",
    "ngay_ghi_nhan". Dùng khi nạp dữ liệu từ file cấu hình/JSON/DB thay vì
    khởi tạo SuKien(...) trực tiếp trong code.
    """
    return SuKien(
        nhom=d["nhom"],
        muc_do=d["muc_do"],
        ghi_chu=d.get("ghi_chu"),
        ngay_ghi_nhan=d.get("ngay_ghi_nhan"),
    )


def danh_sach_nhan_muc_do_hop_le(nhom: str) -> list[str]:
    """Trả về danh sách các nhãn mức độ hợp lệ của 1 nhóm — hữu ích để hiển
    thị dropdown/lựa chọn trên UI khi đội phân tích cập nhật sự kiện thủ công."""
    if nhom not in CAC_NHOM_HOP_LE:
        raise NhomSuKienKhongHopLeError(
            f"Nhóm '{nhom}' không hợp lệ. Các nhóm hợp lệ: {CAC_NHOM_HOP_LE}"
        )
    return list(_BANG_THEO_NHOM[nhom].keys())
