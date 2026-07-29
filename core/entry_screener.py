"""
entry_screener.py
====================
[Bổ sung — Module Rà soát Danh sách Cổ phiếu Vào Lệnh Ngắn Hạn]

Lớp TỔNG HỢP/LỌC trên nền `pattern_detector.py` + `stock_signal_engine.py`
đã có — KHÔNG viết lại logic gốc, chỉ bổ sung:
    1. Xếp hạng ưu tiên theo vị trí giá so với EMA200.
    2. Cờ lọc bổ sung cho mô hình tích lũy dài hạn / dao động tắt dần
       (dựa trên kết quả `pattern_detector.detect_narrowing_pattern()` đã có).
    3. Bộ lọc chọn tiêu chí kết hợp (OR), quét toàn bộ watchlist.
    4. Nâng cấp phân kỳ tăng 2 điểm -> 3 điểm cho độ tin cậy cao hơn
       (dùng khi thị trường Downtrend, sau đợt giảm sâu).

KHÔNG dùng độc lập để tự động đặt lệnh — đây là công cụ RÀ SOÁT/LỌC danh
sách chờ, kết quả cần đối chiếu tiếp với
`stock_signal_engine.evaluate_stock_signal()` trước khi coi là tín hiệu
vào lệnh đầy đủ.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


class InvalidEntryScreenerError(ValueError):
    """Dữ liệu đầu vào không hợp lệ cho module rà soát danh sách vào lệnh."""


TIEU_CHI_KHA_DUNG = {
    "dieu_kien_nen_ema200": "Giá trên EMA200 hoặc trong ±10%",
    "tich_luy_dai_han": "Tích lũy ≥60 phiên, biên độ <5%",
    "dao_dong_tat_dan": "Mô hình thu hẹp biên độ, sắp breakout",
    "volume_breakout": "Khối lượng phiên breakout ≥1.5x TB20",
}


# ==============================================================================
# MỤC 1 — XẾP HẠNG ƯU TIÊN THEO VỊ TRÍ SO VỚI EMA200
# ==============================================================================

def xep_hang_uu_tien_theo_ema200(gia_dong_cua: float, ema200: Optional[float]) -> dict:
    """Xếp hạng ưu tiên vào lệnh dựa trên độ lệch giá so với EMA200 (mục 1
    tài liệu): trên EMA200 -> ưu tiên CAO, trong -10%..0% -> ưu tiên TRUNG
    BÌNH (có thể đang pullback), dưới -10% -> KHÔNG ĐẠT.

    `ema200=None` (mã chưa đủ dữ liệu tính EMA200) -> trả về KHÔNG ĐẠT,
    không báo lỗi (an toàn, loại khỏi danh sách chờ).
    """
    if ema200 is None:
        return {"do_lech_ema200_pct": None, "xep_hang_uu_tien": "KHONG_DAT"}
    if ema200 <= 0:
        raise InvalidEntryScreenerError("ema200 phải > 0.")

    do_lech = (gia_dong_cua - ema200) / ema200 * 100
    if do_lech >= 0:
        hang = "UU_TIEN_CAO"
    elif do_lech >= -10.0:
        hang = "UU_TIEN_TRUNG_BINH"
    else:
        hang = "KHONG_DAT"

    return {"do_lech_ema200_pct": do_lech, "xep_hang_uu_tien": hang}


# ==============================================================================
# MỤC 2 — TÍCH LŨY DÀI HẠN, BIÊN ĐỘ HẸP (≥60 phiên, dao động <5%)
# ==============================================================================

def kiem_tra_tich_luy_dai_han(
    df: pd.DataFrame, lookback: int = 60, max_range_pct: float = 5.0
) -> dict:
    """Kiểm tra tiêu chí tích lũy dài hạn (mục 2 tài liệu) — dạng ĐẶC BIỆT,
    tin cậy CAO NHẤT của mô hình thu hẹp biên độ đã có trong
    `pattern_detector.py` (không phải mô hình tách biệt).
    """
    if len(df) < lookback:
        return {
            "dat": False,
            "ly_do": f"Chưa đủ {lookback} phiên dữ liệu (chỉ có {len(df)}).",
        }

    recent = df.tail(lookback)
    gia_cao_nhat = recent["high"].max()
    gia_thap_nhat = recent["low"].min()
    gia_trung_binh = recent["close"].mean()

    if gia_trung_binh <= 0:
        raise InvalidEntryScreenerError("Giá trung bình phải > 0.")

    bien_do_pct = (gia_cao_nhat - gia_thap_nhat) / gia_trung_binh * 100
    dat = bool(bien_do_pct < max_range_pct)

    return {"dat": dat, "bien_do_dao_dong_pct": bien_do_pct, "so_phien": lookback}


# ==============================================================================
# MỤC 3 — MÔ HÌNH DAO ĐỘNG TẮT DẦN (SẮP BREAKOUT)
# ==============================================================================

def kiem_tra_sap_breakout(
    pattern_result: Optional[dict],
    so_doan_toi_thieu: int = 3,
    bien_do_cuoi_toi_da_pct: float = 5.0,
) -> dict:
    """Kiểm tra điều kiện lọc bổ sung (mục 3 tài liệu) trên kết quả ĐÃ CÓ
    SẴN từ `pattern_detector.detect_narrowing_pattern()` — chỉ gắn cờ
    "sắp breakout" nếu có đủ số đoạn thu hẹp tối thiểu VÀ đoạn cuối cùng
    đã đủ hẹp.
    """
    if not pattern_result:
        return {
            "sap_breakout": False,
            "ly_do": "Không có mô hình thu hẹp biên độ nào được phát hiện.",
        }

    segments = pattern_result.get("segments", [])
    if len(segments) < so_doan_toi_thieu:
        return {
            "sap_breakout": False,
            "ly_do": f"Chưa đủ {so_doan_toi_thieu} đoạn để xác nhận (chỉ có {len(segments)}).",
        }

    bien_do_doan_cuoi = segments[-1]["amplitude_pct"]
    sap_breakout = bien_do_doan_cuoi <= bien_do_cuoi_toi_da_pct

    return {
        "sap_breakout": sap_breakout,
        "bien_do_doan_cuoi_pct": bien_do_doan_cuoi,
        "so_doan": len(segments),
        "chuoi_bien_do_pct": [s["amplitude_pct"] for s in segments],
    }


# ==============================================================================
# MỤC 6 — PHÂN KỲ TĂNG 3 ĐIỂM (nâng cấp, độ tin cậy CAO)
# ==============================================================================

def detect_bullish_divergence_3_diem(
    df: pd.DataFrame, rsi_period: int = 14, lookback: int = 120, swing_order: int = 3,
) -> dict:
    """Mở rộng `market_breadth.detect_bullish_divergence()` — kiểm tra
    thêm đáy swing thứ 3 (trước 2 đáy gần nhất) để xác nhận chuỗi phân kỳ
    LIÊN TỤC (mục 6 tài liệu).

    - Chỉ 2 đáy gần nhất thỏa phân kỳ (giá thấp hơn + RSI cao hơn) ->
      `do_tin_cay="TRUNG_BINH"`.
    - CẢ 2 cặp đáy liên tiếp (đáy 1->2 VÀ đáy 2->3) đều thỏa phân kỳ ->
      `do_tin_cay="CAO"`.
    """
    from core.indicators import calculate_rsi
    from core.market_breadth import _find_local_minima_indices

    min_rows_needed = rsi_period + swing_order * 2 + 1
    if len(df) < min_rows_needed:
        return {
            "detected": False, "do_tin_cay": None,
            "reason": f"Cần tối thiểu {min_rows_needed} phiên dữ liệu, chỉ có {len(df)}.",
        }

    rsi = calculate_rsi(df, period=rsi_period)
    recent_df = df.tail(lookback).reset_index(drop=True)
    recent_rsi = rsi.tail(lookback).reset_index(drop=True)

    lows = recent_df["low"].tolist()
    swing_indices = _find_local_minima_indices(lows, order=swing_order)

    if len(swing_indices) < 2:
        return {
            "detected": False, "do_tin_cay": None,
            "reason": "Không đủ 2 đáy swing đã xác nhận để so sánh phân kỳ.",
        }

    idx1, idx2 = swing_indices[-2], swing_indices[-1]
    price1, price2 = lows[idx1], lows[idx2]
    rsi1, rsi2 = recent_rsi.iloc[idx1], recent_rsi.iloc[idx2]

    if pd.isna(rsi1) or pd.isna(rsi2):
        return {
            "detected": False, "do_tin_cay": None,
            "reason": "RSI chưa đủ dữ liệu tại (các) điểm đáy swing được phát hiện.",
        }

    phan_ky_doan_1 = price2 < price1 and rsi2 > rsi1
    if not phan_ky_doan_1:
        return {
            "detected": False, "do_tin_cay": None,
            "price_low_1": float(price1), "price_low_2": float(price2),
            "rsi_low_1": float(rsi1), "rsi_low_2": float(rsi2),
        }

    ket_qua = {
        "detected": True, "do_tin_cay": "TRUNG_BINH",
        "price_low_1": float(price1), "price_low_2": float(price2),
        "rsi_low_1": float(rsi1), "rsi_low_2": float(rsi2),
    }

    if len(swing_indices) >= 3:
        idx0 = swing_indices[-3]
        price0, rsi0 = lows[idx0], recent_rsi.iloc[idx0]
        if not pd.isna(rsi0):
            phan_ky_doan_2 = price1 < price0 and rsi1 > rsi0
            if phan_ky_doan_2:
                ket_qua["do_tin_cay"] = "CAO"
                ket_qua["price_low_0"] = float(price0)
                ket_qua["rsi_low_0"] = float(rsi0)

    return ket_qua


# ==============================================================================
# MỤC 5 — BỘ LỌC CHÍNH: QUÉT DANH SÁCH CHỜ
# ==============================================================================

def quet_mot_ma(
    symbol: str,
    df: pd.DataFrame,
    ema200: Optional[float],
    pattern_result: Optional[dict],
    resistance_level: Optional[float],
    volume_ma20: Optional[float],
    tieu_chi_da_chon: list[str],
) -> Optional[dict]:
    """Kiểm tra MỘT mã theo các tiêu chí đã chọn (OR — đạt ít nhất 1 tiêu
    chí là được đưa vào danh sách chờ). Trả về `None` nếu không đạt tiêu
    chí nào trong số đã chọn.
    """
    if df is None or df.empty:
        return None

    close = float(df["close"].iloc[-1])
    xep_hang = xep_hang_uu_tien_theo_ema200(close, ema200)

    tieu_chi_dat: list[str] = []
    sap_breakout = False
    mau_hinh_kich_hoat = None
    do_tin_cay_mau_hinh = None

    if "dieu_kien_nen_ema200" in tieu_chi_da_chon and xep_hang["xep_hang_uu_tien"] != "KHONG_DAT":
        tieu_chi_dat.append("dieu_kien_nen_ema200")

    if "tich_luy_dai_han" in tieu_chi_da_chon:
        tich_luy = kiem_tra_tich_luy_dai_han(df)
        if tich_luy["dat"]:
            tieu_chi_dat.append("tich_luy_dai_han")

    if "dao_dong_tat_dan" in tieu_chi_da_chon:
        breakout_check = kiem_tra_sap_breakout(pattern_result)
        if breakout_check["sap_breakout"]:
            tieu_chi_dat.append("dao_dong_tat_dan")
            sap_breakout = True

    if "volume_breakout" in tieu_chi_da_chon and resistance_level is not None and volume_ma20 is not None:
        from core.stock_signal_engine import check_breakout_pattern
        if check_breakout_pattern(df, resistance_level, volume_ma20):
            tieu_chi_dat.append("volume_breakout")
            mau_hinh_kich_hoat = "BREAKOUT"
            do_tin_cay_mau_hinh = "CAO"

    if not tieu_chi_dat:
        return None

    return {
        "ma": symbol,
        "xep_hang_uu_tien": xep_hang["xep_hang_uu_tien"],
        "do_lech_ema200_pct": (
            round(xep_hang["do_lech_ema200_pct"], 2)
            if xep_hang["do_lech_ema200_pct"] is not None else None
        ),
        "tieu_chi_dat": tieu_chi_dat,
        "sap_breakout": sap_breakout,
        "mau_hinh_kich_hoat": mau_hinh_kich_hoat,
        "do_tin_cay_mau_hinh": do_tin_cay_mau_hinh,
    }


def quet_danh_sach_cho(danh_sach_ma_info: list[dict], tieu_chi_da_chon: list[str]) -> dict:
    """Quét TOÀN BỘ watchlist/thị trường theo các tiêu chí đã chọn (mục 5
    tài liệu), trả về danh sách mã ĐẠT, xếp hạng theo `xep_hang_uu_tien`
    giảm dần (CAO -> TRUNG_BÌNH -> KHÔNG_ĐẠT).

    `danh_sach_ma_info`: list[dict], mỗi phần tử cần có:
        {"symbol": str, "df": pd.DataFrame,
         "ema200": Optional[float], "pattern_result": Optional[dict],
         "resistance_level": Optional[float], "volume_ma20": Optional[float]}
    """
    if not tieu_chi_da_chon:
        raise InvalidEntryScreenerError("tieu_chi_da_chon không được rỗng.")

    invalid = set(tieu_chi_da_chon) - set(TIEU_CHI_KHA_DUNG.keys())
    if invalid:
        raise InvalidEntryScreenerError(
            f"Tiêu chí không hợp lệ: {sorted(invalid)}. Cần một trong "
            f"{sorted(TIEU_CHI_KHA_DUNG.keys())}."
        )

    ket_qua = []
    for info in danh_sach_ma_info:
        r = quet_mot_ma(
            symbol=info["symbol"], df=info["df"], ema200=info.get("ema200"),
            pattern_result=info.get("pattern_result"),
            resistance_level=info.get("resistance_level"),
            volume_ma20=info.get("volume_ma20"),
            tieu_chi_da_chon=tieu_chi_da_chon,
        )
        if r is not None:
            ket_qua.append(r)

    thu_tu_uu_tien = {"UU_TIEN_CAO": 0, "UU_TIEN_TRUNG_BINH": 1, "KHONG_DAT": 2}
    ket_qua_sorted = sorted(
        ket_qua, key=lambda r: thu_tu_uu_tien.get(r["xep_hang_uu_tien"], 3)
    )

    return {
        "tieu_chi_da_chon": tieu_chi_da_chon,
        "danh_sach_ma": ket_qua_sorted,
        "tong_so_ma_dat": len(ket_qua_sorted),
        "tong_so_ma_da_quet": len(danh_sach_ma_info),
        "ghi_chu": (
            "Danh sách CHỜ tham khảo — cần đối chiếu Module Tín hiệu Mua/Bán "
            "trước khi ra quyết định vào lệnh cụ thể."
        ),
    }
