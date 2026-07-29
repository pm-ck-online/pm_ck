"""
short_term_signal.py
======================
[Bổ sung — Module Tiêu chí Ngắn hạn]

Lớp tín hiệu NGẮN HẠN dựa trên độ lệch giá so với MA20, áp dụng cho cả
VN-Index (toàn thị trường) và từng cổ phiếu riêng lẻ:
    1. Cảnh báo quá mua ngắn hạn (VN-Index + từng mã).
    2. Thống kê xác suất điều chỉnh sau N phiên (event study / backtest).
    3. Tín hiệu "bắt cá hồi" (mean-reversion bounce) sau giảm mạnh.

Đây là lớp tín hiệu BỔ SUNG cho `stock_signal_engine.py` — KHÔNG dùng độc
lập để tự động đặt lệnh, chỉ là cảnh báo/thống kê tham khảo.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


class InvalidShortTermSignalError(ValueError):
    """Dữ liệu đầu vào không hợp lệ cho module tiêu chí ngắn hạn."""


# ==============================================================================
# MỤC 1 — VN-INDEX VƯỢT XA MA20 (CẢNH BÁO ĐIỀU CHỈNH TOÀN THỊ TRƯỜNG)
# ==============================================================================

def canh_bao_qua_mua_vnindex(gia_dong_cua: float, ma20: float) -> dict:
    """Cảnh báo quá mua ngắn hạn cho VN-Index dựa trên độ lệch so với MA20
    (mục 1 tài liệu): < 2% Bình thường, 2-3% Cảnh báo điều chỉnh, ≥4% Nguy cơ cao.
    """
    if ma20 <= 0:
        raise InvalidShortTermSignalError("ma20 phải > 0.")

    do_lech = (gia_dong_cua - ma20) / ma20 * 100

    if do_lech >= 4.0:
        muc = "NGUY_CO_CAO"
    elif do_lech >= 2.0:
        muc = "CANH_BAO_DIEU_CHINH"
    else:
        muc = "BINH_THUONG"

    return {"do_lech_ma20_pct": do_lech, "muc_canh_bao": muc}


# ==============================================================================
# MỤC 2 — CỔ PHIẾU RIÊNG LẺ VƯỢT XA MA20
# ==============================================================================

def canh_bao_qua_mua_co_phieu(ma_cp: str, gia_dong_cua: float, ma20: float) -> dict:
    """Cảnh báo quá mua ngắn hạn cho MỘT CỔ PHIẾU (mục 2 tài liệu): < 10%
    Bình thường, 10-15% Nguy cơ điều chỉnh, >15% Nguy cơ điều chỉnh cao.
    """
    if ma20 <= 0:
        raise InvalidShortTermSignalError("ma20 phải > 0.")

    do_lech = (gia_dong_cua - ma20) / ma20 * 100

    if do_lech > 15.0:
        muc = "NGUY_CO_CAO"
    elif do_lech >= 10.0:
        muc = "NGUY_CO_DIEU_CHINH"
    else:
        muc = "BINH_THUONG"

    return {"ma": ma_cp, "do_lech_ma20_pct": do_lech, "muc_canh_bao": muc}


# ==============================================================================
# MỤC 3 — THỐNG KÊ XÁC SUẤT ĐIỀU CHỈNH SAU N PHIÊN (Event Study)
# ==============================================================================

def _find_threshold_onset_indices(do_lech: pd.Series, nguong_canh_bao_pct: float) -> list[int]:
    """Tìm các chỉ số mà độ lệch MA20 VỪA VƯỢT ngưỡng (điểm bắt đầu của mỗi
    đợt vượt ngưỡng — tránh đếm trùng nhiều phiên liên tiếp cùng 1 đợt).
    """
    above = do_lech >= nguong_canh_bao_pct
    onset_indices = []
    for i in range(len(above)):
        is_above = bool(above.iloc[i]) if not pd.isna(do_lech.iloc[i]) else False
        was_above = bool(above.iloc[i - 1]) if i > 0 and not pd.isna(do_lech.iloc[i - 1]) else False
        if is_above and not was_above:
            onset_indices.append(i)
    return onset_indices


def thong_ke_xac_suat_dieu_chinh(
    lich_su_gia: pd.Series,
    lich_su_ma20: pd.Series,
    nguong_canh_bao_pct: float,
    cac_khung_ngay: Optional[list[int]] = None,
    nguong_dieu_chinh_pct: float = 3.0,
) -> dict:
    """Thống kê xác suất điều chỉnh sau N phiên, dựa trên toàn bộ LỊCH SỬ
    những lần độ lệch MA20 vừa vượt `nguong_canh_bao_pct` (mục 3 tài liệu).

    Trả về:
        {"tong_so_su_kien": int,
         "xac_suat_theo_khung_ngay": {N: {"xac_suat_pct", "muc_dieu_chinh_tb_pct",
                                           "so_phien_tb_toi_day", "so_su_kien_hop_le"}}}
    """
    cac_khung_ngay = cac_khung_ngay or [5, 10, 20]

    if len(lich_su_gia) != len(lich_su_ma20):
        raise InvalidShortTermSignalError("lich_su_gia và lich_su_ma20 phải cùng độ dài.")
    if len(lich_su_gia) == 0:
        raise InvalidShortTermSignalError("lich_su_gia không được rỗng.")

    lich_su_gia = lich_su_gia.reset_index(drop=True)
    lich_su_ma20 = lich_su_ma20.reset_index(drop=True)
    do_lech = (lich_su_gia - lich_su_ma20) / lich_su_ma20 * 100

    event_indices = _find_threshold_onset_indices(do_lech, nguong_canh_bao_pct)

    ket_qua_theo_khung: dict[int, dict] = {}
    for N in cac_khung_ngay:
        so_lan_dieu_chinh = 0
        muc_dieu_chinh_list: list[float] = []
        so_phien_toi_day_list: list[int] = []
        so_su_kien_hop_le = 0

        for idx in event_indices:
            if idx + N >= len(lich_su_gia):
                continue  # không đủ dữ liệu tương lai để xét khung N phiên này
            so_su_kien_hop_le += 1

            gia_goc = lich_su_gia.iloc[idx]
            cua_so_tuong_lai = lich_su_gia.iloc[idx + 1: idx + 1 + N]
            gia_thap_nhat = cua_so_tuong_lai.min()
            muc_giam_pct = (gia_goc - gia_thap_nhat) / gia_goc * 100

            if muc_giam_pct >= nguong_dieu_chinh_pct:
                so_lan_dieu_chinh += 1
                muc_dieu_chinh_list.append(muc_giam_pct)
                vi_tri_thap_nhat = cua_so_tuong_lai.values.argmin() + 1
                so_phien_toi_day_list.append(int(vi_tri_thap_nhat))

        if so_su_kien_hop_le == 0:
            ket_qua_theo_khung[N] = {
                "xac_suat_pct": None, "muc_dieu_chinh_tb_pct": None,
                "so_phien_tb_toi_day": None, "so_su_kien_hop_le": 0,
            }
            continue

        ket_qua_theo_khung[N] = {
            "xac_suat_pct": round(so_lan_dieu_chinh / so_su_kien_hop_le * 100, 1),
            "muc_dieu_chinh_tb_pct": (
                round(sum(muc_dieu_chinh_list) / len(muc_dieu_chinh_list), 2)
                if muc_dieu_chinh_list else None
            ),
            "so_phien_tb_toi_day": (
                round(sum(so_phien_toi_day_list) / len(so_phien_toi_day_list), 1)
                if so_phien_toi_day_list else None
            ),
            "so_su_kien_hop_le": so_su_kien_hop_le,
        }

    return {
        "tong_so_su_kien": len(event_indices),
        "xac_suat_theo_khung_ngay": ket_qua_theo_khung,
    }


# ==============================================================================
# MỤC 4 — "BẮT CÁ HỒI" (Mean-Reversion Bounce)
# ==============================================================================

NGANH_UU_TIEN_BAT_CA_HOI = ["Ngân hàng", "Chứng khoán", "Thép"]
RO_MA_UU_TIEN_BAT_CA_HOI = "VN30"


def kiem_tra_tin_hieu_bat_ca_hoi(
    gia_hien_tai: float,
    lich_su_gia_40_phien,
    macro_score: Optional[float] = None,
) -> dict:
    """Kiểm tra tín hiệu bắt cá hồi sau giảm mạnh (mục 4 tài liệu), có áp
    dụng điều kiện PHỦ QUYẾT (mục 4.4): không kích hoạt nếu Macro Score
    < -1.0 hoặc mức giảm từ đỉnh > 20%.
    """
    if len(lich_su_gia_40_phien) == 0:
        raise InvalidShortTermSignalError("lich_su_gia_40_phien không được rỗng.")

    dinh_cao_nhat = max(lich_su_gia_40_phien)
    if dinh_cao_nhat <= 0:
        raise InvalidShortTermSignalError("Đỉnh cao nhất phải > 0.")

    muc_giam_pct = (dinh_cao_nhat - gia_hien_tai) / dinh_cao_nhat * 100
    dieu_kien_gia_thoa_man = 10.0 <= muc_giam_pct <= 15.0

    phu_quyet_ly_do: list[str] = []
    if macro_score is not None and macro_score < -1.0:
        phu_quyet_ly_do.append(
            "Macro Score tiêu cực mạnh (<-1.0) — đợt giảm có thể còn tiếp diễn "
            "do yếu tố vĩ mô, không phải nhịp điều chỉnh kỹ thuật thông thường."
        )
    if muc_giam_pct > 20.0:
        phu_quyet_ly_do.append(
            "Mức giảm từ đỉnh vượt 20% — đã chuyển sang downtrend xác nhận, "
            "không còn là bối cảnh bắt cá hồi ngắn hạn."
        )

    kich_hoat = bool(dieu_kien_gia_thoa_man and not phu_quyet_ly_do)

    return {
        "kich_hoat": kich_hoat,
        "muc_giam_tu_dinh_40_phien_pct": round(muc_giam_pct, 2),
        "dinh_cao_nhat_40_phien": dinh_cao_nhat,
        "nganh_uu_tien": list(NGANH_UU_TIEN_BAT_CA_HOI) if kich_hoat else [],
        "ro_ma_uu_tien": RO_MA_UU_TIEN_BAT_CA_HOI if kich_hoat else None,
        "phu_quyet_ly_do": phu_quyet_ly_do,
        "ghi_chu": (
            "Tín hiệu bắt cá hồi kỹ thuật — CHỈ áp dụng cho vị thế NGẮN HẠN, "
            "cần kết hợp thêm xác nhận từ RSI/Volume trước khi vào lệnh."
        ) if kich_hoat else None,
    }


# ==============================================================================
# HÀM TỔNG HỢP — output theo đúng cấu trúc mục 5 tài liệu
# ==============================================================================

def build_short_term_signal_report(
    vnindex_close: float,
    vnindex_ma20: float,
    vnindex_history_close: pd.Series,
    vnindex_history_ma20: pd.Series,
    vnindex_history_40d,
    stock_snapshots: Optional[list[dict]] = None,
    macro_score: Optional[float] = None,
    danh_gia_date: Optional[str] = None,
    nguong_canh_bao_vnindex_pct: float = 2.0,
) -> dict:
    """Tổng hợp báo cáo tiêu chí ngắn hạn đầy đủ (mục 5 tài liệu).

    `stock_snapshots`: danh sách [{"ma": str, "close": float, "ma20": float}]
    cho các mã cần kiểm tra quá mua riêng lẻ (mục 2). Chỉ các mã KHÔNG ở
    mức "BINH_THUONG" mới được đưa vào `co_phieu_qua_mua` của output.
    """
    vnindex_canh_bao = canh_bao_qua_mua_vnindex(vnindex_close, vnindex_ma20)
    xac_suat = thong_ke_xac_suat_dieu_chinh(
        vnindex_history_close, vnindex_history_ma20,
        nguong_canh_bao_pct=nguong_canh_bao_vnindex_pct,
    )

    tin_hieu_bat_ca_hoi = kiem_tra_tin_hieu_bat_ca_hoi(
        vnindex_close, vnindex_history_40d, macro_score=macro_score,
    )

    co_phieu_qua_mua = []
    canh_bao: list[str] = []
    for snap in (stock_snapshots or []):
        result = canh_bao_qua_mua_co_phieu(snap["ma"], snap["close"], snap["ma20"])
        if result["muc_canh_bao"] != "BINH_THUONG":
            co_phieu_qua_mua.append(result)
            if result["muc_canh_bao"] == "NGUY_CO_CAO":
                canh_bao.append(f"[{snap['ma']}] Độ lệch MA20 vượt 15% — nguy cơ điều chỉnh cao.")

    if vnindex_canh_bao["muc_canh_bao"] == "NGUY_CO_CAO":
        canh_bao.append("VN-Index vượt MA20 ≥4% — nguy cơ điều chỉnh cao toàn thị trường.")

    return {
        "ngay_danh_gia": danh_gia_date,
        "vnindex": {
            "do_lech_ma20_pct": round(vnindex_canh_bao["do_lech_ma20_pct"], 2),
            "muc_canh_bao": vnindex_canh_bao["muc_canh_bao"],
            "xac_suat_dieu_chinh": {
                "tong_so_su_kien_lich_su": xac_suat["tong_so_su_kien"],
                "theo_khung_ngay": xac_suat["xac_suat_theo_khung_ngay"],
            },
        },
        "tin_hieu_bat_ca_hoi": tin_hieu_bat_ca_hoi,
        "co_phieu_qua_mua": co_phieu_qua_mua,
        "canh_bao": canh_bao,
        "ghi_chu": (
            "Chỉ tiêu ngắn hạn mang tính tham khảo thống kê, không phải "
            "khuyến nghị đầu tư cá nhân hóa."
        ),
    }
