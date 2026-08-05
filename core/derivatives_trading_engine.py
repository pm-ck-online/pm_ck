"""
derivatives_trading_engine.py
================================
Công thức giao dịch Hợp đồng tương lai chỉ số VN30 (VN30F1M) — vào lệnh
(entry range), phân bổ vốn (số hợp đồng tối ưu, ràng buộc KÉP: rủi ro
2% NAV VÀ trần ký quỹ), và tính R:R (Risk:Reward) cho từng kịch bản.

KHÁC BIỆT QUAN TRỌNG so với `stock_signal_engine.py`/`capital_allocator.py`
(dành cho cổ phiếu): HĐTL có đòn bẩy qua cơ chế KÝ QUỸ (margin), thanh
toán bù trừ HÀNG NGÀY (mark-to-market), và HỆ SỐ NHÂN HỢP ĐỒNG cố định
100.000đ/điểm — 3 đặc tính này bắt buộc phải có ràng buộc riêng, KHÔNG
tái dùng nguyên vẹn `capital_allocator.py` gốc (vốn thiết kế cho cổ phiếu,
không có khái niệm ký quỹ hay hệ số nhân).

CHỈ HỖ TRỢ TÍNH TOÁN THAM KHẢO — KHÔNG tự động đặt lệnh, không phải
khuyến nghị đầu tư hay đảm bảo lợi nhuận. Phái sinh có đòn bẩy cao, cơ
chế thanh toán bù trừ hàng ngày có thể gây lỗ nhanh hơn nhiều so với cổ
phiếu thường.
"""

from __future__ import annotations

from typing import Optional

# ==============================================================================
# HẰNG SỐ / ÁNH XẠ
# ==============================================================================

# (hệ số cận dưới, hệ số cận trên) nhân với ATR14, CỘNG vào giá tham chiếu.
HE_SO_BIEN_DO_ENTRY = {
    "BREAKOUT_TANG": (0.05, 0.40),
    "BREAKOUT_GIAM": (-0.40, -0.05),
    "MUA_HO_TRO": (-0.20, 0.20),
    "BAN_KHANG_CU": (-0.20, 0.20),
}

HUONG_THEO_TIN_HIEU = {
    "BREAKOUT_TANG": "LONG", "MUA_HO_TRO": "LONG",
    "BREAKOUT_GIAM": "SHORT", "BAN_KHANG_CU": "SHORT",
}

NGUONG_DANH_GIA_RR = [
    (2.0, "TOT"),
    (1.5, "CHAP_NHAN_DUOC"),
    (1.0, "THAP_CAN_XEM_XET_LAI"),
]


class InvalidDerivativesError(ValueError):
    """Dữ liệu đầu vào không hợp lệ cho module giao dịch HĐTL VN30."""


# ==============================================================================
# 1. CÔNG THỨC VÀO LỆNH (Entry Range)
# ==============================================================================

def tinh_khoang_gia_vao_lenh(gia_tham_chieu: float, atr14: float, kieu_tin_hieu: str) -> tuple[float, float]:
    """Tính khoảng giá vào lệnh (điểm chỉ số) theo kiểu tín hiệu, dựa trên
    ATR14 nhân với hệ số biên độ tương ứng (xem `HE_SO_BIEN_DO_ENTRY`).

    Trả về tuple (cận_thấp, cận_cao) — LUÔN theo thứ tự tăng dần bất kể
    hướng LONG/SHORT (VD: BREAKOUT_GIAM có hệ số âm nhưng cận_thấp vẫn
    nhỏ hơn cận_cao về mặt số học).
    """
    if kieu_tin_hieu not in HE_SO_BIEN_DO_ENTRY:
        raise InvalidDerivativesError(
            f"kieu_tin_hieu phải là 1 trong {sorted(HE_SO_BIEN_DO_ENTRY.keys())}."
        )
    if gia_tham_chieu <= 0:
        raise InvalidDerivativesError("gia_tham_chieu phải > 0.")
    if atr14 <= 0:
        raise InvalidDerivativesError("atr14 phải > 0.")

    k_duoi, k_tren = HE_SO_BIEN_DO_ENTRY[kieu_tin_hieu]
    cac_gia = (
        round(gia_tham_chieu + k_duoi * atr14, 1),
        round(gia_tham_chieu + k_tren * atr14, 1),
    )
    return (min(cac_gia), max(cac_gia))


# ==============================================================================
# 2. CÔNG THỨC PHÂN BỔ VỐN — SỐ HỢP ĐỒNG TỐI ƯU (RÀNG BUỘC KÉP)
# ==============================================================================

def tinh_so_hop_dong_toi_uu(
    nav: float,
    gia_vao_toi_da: float,
    gia_cat_lo: float,
    gia_tham_chieu_ky_quy: float,
    ty_le_ky_quy: float = 0.15,
    ty_le_ky_quy_toi_da_nav: float = 0.50,
    rui_ro_moi_lenh_pct: float = 0.02,
    he_so_nhan: float = 100_000,
    gioi_han_lenh_toi_da: int = 500,
) -> dict:
    """Tính số hợp đồng tối ưu = MIN của 3 ràng buộc:
        1. Ngân sách rủi ro (`rui_ro_moi_lenh_pct` × NAV, mặc định 2%)
        2. Trần ký quỹ cho phép (`ty_le_ky_quy_toi_da_nav` × NAV, mặc định 50%)
        3. Giới hạn lệnh theo quy định (`gioi_han_lenh_toi_da`, mặc định 500 HĐ/lệnh)

    Đồng thời trả về `nut_that_gioi_han` — ràng buộc nào đang là "nút
    thắt" (binding constraint), để người dùng hiểu ĐÚNG bản chất giới
    hạn, tránh hiểu nhầm hệ thống tính sai.

    LƯU Ý ĐÃ KIỂM CHỨNG (04/08/2026): với phong cách "lướt trong ngày"
    (cắt lỗ hẹp theo ATR khung giờ), ràng buộc trần ký quỹ THƯỜNG là nút
    thắt; với "giữ theo kịch bản" (cắt lỗ rộng theo ATR ngày), ràng buộc
    rủi ro 2% THƯỜNG là nút thắt — đây là hệ quả tự nhiên của công thức,
    không phải lỗi.
    """
    if nav <= 0:
        raise InvalidDerivativesError("nav phải > 0.")
    if gia_vao_toi_da <= 0 or gia_tham_chieu_ky_quy <= 0:
        raise InvalidDerivativesError("gia_vao_toi_da và gia_tham_chieu_ky_quy phải > 0.")
    if gia_vao_toi_da == gia_cat_lo:
        raise InvalidDerivativesError("gia_cat_lo không được trùng gia_vao_toi_da (rủi ro phải > 0).")
    if not (0 < ty_le_ky_quy < 1):
        raise InvalidDerivativesError("ty_le_ky_quy phải trong khoảng (0, 1).")
    if not (0 < ty_le_ky_quy_toi_da_nav <= 1):
        raise InvalidDerivativesError("ty_le_ky_quy_toi_da_nav phải trong khoảng (0, 1].")
    if not (0 < rui_ro_moi_lenh_pct <= 1):
        raise InvalidDerivativesError("rui_ro_moi_lenh_pct phải trong khoảng (0, 1].")

    rui_ro_diem = abs(gia_vao_toi_da - gia_cat_lo)
    rui_ro_tren_1_hd = rui_ro_diem * he_so_nhan

    ngan_sach_rui_ro = nav * rui_ro_moi_lenh_pct
    so_hd_theo_rui_ro = ngan_sach_rui_ro / rui_ro_tren_1_hd if rui_ro_tren_1_hd > 0 else float("inf")

    gia_tri_danh_nghia_1_hd = gia_tham_chieu_ky_quy * he_so_nhan
    ky_quy_yeu_cau_1_hd = gia_tri_danh_nghia_1_hd * ty_le_ky_quy
    ngan_sach_ky_quy = nav * ty_le_ky_quy_toi_da_nav
    so_hd_theo_ky_quy = ngan_sach_ky_quy / ky_quy_yeu_cau_1_hd if ky_quy_yeu_cau_1_hd > 0 else float("inf")

    cac_rang_buoc = {
        "theo_rui_ro_2pct": so_hd_theo_rui_ro,
        "theo_tran_ky_quy": so_hd_theo_ky_quy,
        "theo_gioi_han_quy_dinh": float(gioi_han_lenh_toi_da),
    }
    nut_that = min(cac_rang_buoc, key=cac_rang_buoc.get)
    so_hop_dong_cuoi = int(min(cac_rang_buoc.values()))

    ket_qua = {
        "so_hop_dong": so_hop_dong_cuoi,
        "rui_ro_tren_1_hd_vnd": round(rui_ro_tren_1_hd),
        "ky_quy_yeu_cau_1_hd_vnd": round(ky_quy_yeu_cau_1_hd),
        "tong_ky_quy_su_dung_vnd": round(ky_quy_yeu_cau_1_hd * so_hop_dong_cuoi),
        "tong_ky_quy_pct_nav": round(ky_quy_yeu_cau_1_hd * so_hop_dong_cuoi / nav * 100, 1),
        "nut_that_gioi_han": nut_that,
        "chi_tiet_cac_rang_buoc": {
            k: (int(v) if v != float("inf") else None) for k, v in cac_rang_buoc.items()
        },
    }

    # Rà soát bắt buộc (mục 7.3 tài liệu gốc): không được để lọt trường
    # hợp vượt trần ký quỹ do làm tròn số hợp đồng — nếu lỡ xảy ra (chỉ
    # có thể do lỗi logic, KHÔNG nên xảy ra với công thức đúng), báo rõ
    # thay vì âm thầm trả về số liệu sai.
    if ket_qua["tong_ky_quy_pct_nav"] > ty_le_ky_quy_toi_da_nav * 100 + 0.05:
        raise InvalidDerivativesError(
            "Lỗi logic nội bộ: tổng ký quỹ sử dụng vượt trần cho phép sau khi làm "
            "tròn số hợp đồng — cần rà soát lại `tinh_so_hop_dong_toi_uu()`."
        )

    return ket_qua


# ==============================================================================
# 3. CÔNG THỨC TÍNH R:R (RISK : REWARD)
# ==============================================================================

def tinh_ty_le_rr(gia_vao: float, gia_cat_lo: float, gia_chot_loi: float, huong: str) -> dict:
    """Tính tỷ lệ R:R (Risk:Reward) cho 1 kịch bản, kèm nhãn đánh giá
    theo ngưỡng chuẩn (xem `NGUONG_DANH_GIA_RR`)."""
    if huong not in ("LONG", "SHORT"):
        raise InvalidDerivativesError('huong phải là "LONG" hoặc "SHORT".')

    if huong == "LONG":
        rui_ro = gia_vao - gia_cat_lo
        loi_nhuan = gia_chot_loi - gia_vao
    else:
        rui_ro = gia_cat_lo - gia_vao
        loi_nhuan = gia_vao - gia_chot_loi

    if rui_ro <= 0:
        raise InvalidDerivativesError("Cắt lỗ không hợp lệ theo hướng lệnh (rủi ro phải > 0).")

    rr = round(loi_nhuan / rui_ro, 2)

    danh_gia = "KHONG_NEN_VAO_LENH"
    for nguong, nhan in NGUONG_DANH_GIA_RR:
        if rr >= nguong:
            danh_gia = nhan
            break

    return {"rr": rr, "danh_gia": danh_gia}


# ==============================================================================
# 4. HÀM CHÍNH — GHÉP TOÀN BỘ PIPELINE
# ==============================================================================

def phan_tich_lenh_hdtl_vn30(
    nav: float,
    phong_cach: str,
    kieu_tin_hieu: str,
    gia_tham_chieu: float,
    atr14: float,
    gia_cat_lo: float,
    gia_chot_loi_du_kien: Optional[float] = None,
    ty_le_ky_quy: float = 0.15,
    ty_le_ky_quy_toi_da_nav: float = 0.50,
    rui_ro_moi_lenh_pct: float = 0.02,
) -> dict:
    """Hàm chính: ghép toàn bộ 3 khối công thức (entry -> phân bổ vốn ->
    R:R) thành 1 kết quả phân tích đầy đủ cho 1 lệnh HĐTL VN30.

    CHỈ HỖ TRỢ TÍNH TOÁN THAM KHẢO — không tự động đặt lệnh, việc đặt
    lệnh thực tế luôn do người dùng xác nhận thủ công.
    """
    if phong_cach not in ("luot_trong_ngay", "giu_theo_kich_ban"):
        raise InvalidDerivativesError(
            'phong_cach phải là "luot_trong_ngay" hoặc "giu_theo_kich_ban".'
        )
    if kieu_tin_hieu not in HUONG_THEO_TIN_HIEU:
        raise InvalidDerivativesError(
            f"kieu_tin_hieu phải là 1 trong {sorted(HUONG_THEO_TIN_HIEU.keys())}."
        )

    huong = HUONG_THEO_TIN_HIEU[kieu_tin_hieu]
    khoang_entry = tinh_khoang_gia_vao_lenh(gia_tham_chieu, atr14, kieu_tin_hieu)
    gia_vao_toi_da = max(khoang_entry) if huong == "LONG" else min(khoang_entry)

    phan_bo = tinh_so_hop_dong_toi_uu(
        nav=nav, gia_vao_toi_da=gia_vao_toi_da, gia_cat_lo=gia_cat_lo,
        gia_tham_chieu_ky_quy=gia_vao_toi_da, ty_le_ky_quy=ty_le_ky_quy,
        ty_le_ky_quy_toi_da_nav=ty_le_ky_quy_toi_da_nav,
        rui_ro_moi_lenh_pct=rui_ro_moi_lenh_pct,
    )

    ket_qua_rr = None
    if gia_chot_loi_du_kien is not None:
        gia_vao_trung_binh = sum(khoang_entry) / 2
        ket_qua_rr = tinh_ty_le_rr(gia_vao_trung_binh, gia_cat_lo, gia_chot_loi_du_kien, huong)

    canh_bao = []
    if ket_qua_rr and ket_qua_rr["danh_gia"] in ("THAP_CAN_XEM_XET_LAI", "KHONG_NEN_VAO_LENH"):
        canh_bao.append(
            "R:R dưới ngưỡng chấp nhận được — cân nhắc lại mục tiêu chốt lời hoặc bỏ qua tín hiệu này."
        )

    return {
        "huong": huong,
        "kieu_tin_hieu": kieu_tin_hieu,
        "phong_cach": phong_cach,
        "khoang_gia_vao_lenh": khoang_entry,
        "gia_cat_lo": gia_cat_lo,
        **phan_bo,
        "ty_le_rr": ket_qua_rr,
        "canh_bao": canh_bao,
        "canh_bao_phap_ly": (
            "Công cụ TÍNH TOÁN THAM KHẢO — KHÔNG tự động đặt lệnh, không phải khuyến "
            "nghị đầu tư hay đảm bảo lợi nhuận. Phái sinh có đòn bẩy cao, cơ chế thanh "
            "toán bù trừ hàng ngày (mark-to-market) có thể gây lỗ nhanh hơn nhiều so "
            "với cổ phiếu thường."
        ),
    }
