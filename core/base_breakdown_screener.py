"""
base_breakdown_screener.py
=============================
[Module 7 — bộ lọc thời gian thực, độc lập với Module 6]

Quét THỜI ĐIỂM HIỆN TẠI trên toàn bộ watchlist để tìm các mã ĐANG thỏa
đồng thời 3 tiêu chí cốt lõi:
    1. Giá đã giảm đủ mạnh tính từ mức PIVOT của vùng nền (base/consolidation
       zone) gần nhất bị đứt gãy — không phải giảm % so với N phiên cố định.
    2. RSI(14) hiện tại < ngưỡng quá bán.
    3. Khối lượng phiên hiện tại tăng đột biến so với trung bình 20 phiên.

Đây là bộ lọc RÚT GỌN có chủ đích — chỉ giữ 3 yếu tố cốt lõi để chạy nhanh
trên toàn bộ watchlist mỗi phiên. KHÔNG thay thế cho:
    - `core.stock_character_classifier` (Module 5 — tính cách giao dịch)
    - `core.historical_recovery_probability` (Module 6 — xác suất phục hồi
      lịch sử) — mã lọt qua bộ lọc này NÊN được phân tích tiếp bằng cả 2
      module trên trước khi cân nhắc bất kỳ quyết định nào.

CHỈ SÀNG LỌC KỸ THUẬT — KHÔNG phải tín hiệu mua/bán hay khuyến nghị đầu tư.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.indicators import calculate_bollinger_bands, calculate_rsi

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


class InvalidBaseBreakdownError(ValueError):
    """Dữ liệu đầu vào không hợp lệ cho module sàng lọc đứt gãy vùng nền."""


def _validate_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise InvalidBaseBreakdownError("df rỗng hoặc None.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise InvalidBaseBreakdownError(f"df thiếu cột bắt buộc: {missing}.")


# ==============================================================================
# BƯỚC 1 — XÁC ĐỊNH "VÙNG NỀN" (Base/Consolidation Zone)
# ==============================================================================

def xac_dinh_vung_nen(df: pd.DataFrame, lookback: int = 60, min_ngay: int = 10) -> Optional[dict]:
    """Quét `lookback` phiên gần nhất, tìm cửa sổ con có Bollinger Band
    Width trung bình THẤP NHẤT kéo dài >= `min_ngay` phiên — đây chính là
    "vùng nền tích lũy" gần nhất trước khi giá đứt gãy.

    Trả về None nếu không tìm được vùng nền hợp lệ (mã luôn biến động
    mạnh, không có giai đoạn tích lũy rõ ràng, hoặc chưa đủ dữ liệu).
    """
    _validate_df(df)
    if len(df) < min_ngay:
        return None

    df_scope = df.iloc[-lookback:].reset_index(drop=True)
    upper, middle, lower = calculate_bollinger_bands(df_scope, period=20)
    band_width_pct = (upper - lower) / middle * 100

    best_window: Optional[tuple[int, int]] = None
    best_avg_width = np.inf
    n = len(band_width_pct)
    for start in range(n - min_ngay + 1):
        for end in range(start + min_ngay, n + 1):
            cua_so = band_width_pct.iloc[start:end]
            if cua_so.isna().any():
                continue
            avg_width = cua_so.mean()
            # Ưu tiên cửa sổ RỘNG HƠN (nhiều phiên tích lũy hơn) nếu độ
            # rộng trung bình xấp xỉ nhau (trong khoảng 10%) để tránh chọn
            # cửa sổ quá ngắn/nhiễu ngẫu nhiên.
            current_best_len = (best_window[1] - best_window[0]) if best_window else 0
            if avg_width < best_avg_width * 0.9 or (
                avg_width < best_avg_width * 1.1 and (end - start) > current_best_len
            ):
                best_avg_width = avg_width
                best_window = (start, end)

    if best_window is None:
        return None

    start, end = best_window
    vung = df_scope.iloc[start:end]
    # Vị trí TUYỆT ĐỐI của điểm bắt đầu vùng nền trong `df` gốc (không
    # phải trong df_scope đã cắt) — dùng offset giữa độ dài df và df_scope.
    offset = len(df) - len(df_scope)
    chi_so_bat_dau_tuyet_doi = offset + start

    return {
        "chi_so_bat_dau": chi_so_bat_dau_tuyet_doi,
        "so_phien_tich_luy": end - start,
        "gia_pivot_ho_tro": float(vung["low"].min()),       # mức hỗ trợ = đáy thấp nhất trong vùng nền
        "gia_pivot_dong_cua_thap_nhat": float(vung["close"].min()),
        "band_width_trung_binh_pct": round(float(best_avg_width), 2),
    }


# ==============================================================================
# BƯỚC 2 — XÁC ĐỊNH "ĐIỂM GÃY" VÀ TÍNH % GIẢM TỪ PIVOT
# ==============================================================================

def tinh_muc_giam_tu_diem_gay(df: pd.DataFrame, vung_nen: dict) -> dict:
    """% giảm được tính từ MỨC PIVOT (đáy vùng nền) đến giá đóng cửa HIỆN
    TẠI (phiên gần nhất) — đúng yêu cầu "tính từ vùng giá gãy trong 60
    phiên trước đó", không phải giảm so với N phiên cố định.
    """
    _validate_df(df)
    pivot = vung_nen["gia_pivot_ho_tro"]
    gia_hien_tai = float(df["close"].iloc[-1])

    pct_giam_tu_pivot = (pivot - gia_hien_tai) / pivot * 100 if pivot else 0.0

    return {
        "gia_pivot_ho_tro": pivot,
        "gia_hien_tai": gia_hien_tai,
        "pct_giam_tu_pivot": round(pct_giam_tu_pivot, 2),
        "da_dut_gay": gia_hien_tai < pivot,
    }


# ==============================================================================
# BƯỚC 3 — RSI VÀ VOLUME (tái sử dụng calculate_rsi đã có sẵn)
# ==============================================================================

def kiem_tra_rsi_va_volume(df: pd.DataFrame, nguong_rsi: float, nguong_volume_ratio: float) -> dict:
    _validate_df(df)
    rsi_series = calculate_rsi(df, period=14)
    rsi_hien_tai = float(rsi_series.iloc[-1]) if not rsi_series.empty and pd.notna(rsi_series.iloc[-1]) else None

    volume_ma20 = df["volume"].rolling(20).mean().iloc[-1]
    volume_hien_tai = df["volume"].iloc[-1]
    volume_ratio = (
        float(volume_hien_tai / volume_ma20) if volume_ma20 not in (0, None) and pd.notna(volume_ma20) else None
    )

    return {
        "rsi_hien_tai": round(rsi_hien_tai, 1) if rsi_hien_tai is not None else None,
        "dat_rsi": rsi_hien_tai is not None and rsi_hien_tai < nguong_rsi,
        "volume_ratio": round(volume_ratio, 2) if volume_ratio is not None else None,
        "dat_volume": volume_ratio is not None and volume_ratio >= nguong_volume_ratio,
    }


# ==============================================================================
# HÀM CHÍNH — QUÉT TOÀN BỘ WATCHLIST
# ==============================================================================

def quet_co_phieu_dut_gay_qua_ban(
    danh_sach_ma: list[str],
    lay_ohlcv_fn: Callable[[str], Optional[pd.DataFrame]],
    lookback_vung_nen: int = 60,
    min_ngay_vung_nen: int = 10,
    nguong_giam_toi_thieu_pct: float = 15.0,
    nguong_rsi: float = 30,
    nguong_volume_ratio: float = 1.5,
) -> pd.DataFrame:
    """Quét TOÀN BỘ watchlist, trả về DataFrame CHỈ gồm các mã thỏa ĐỒNG
    THỜI cả 3 tiêu chí. Không dừng cả vòng lặp nếu 1 mã lỗi/thiếu dữ liệu.
    """
    rows = []
    for ma in danh_sach_ma:
        try:
            df = lay_ohlcv_fn(ma)
            if df is None or len(df) < lookback_vung_nen + 20:
                continue  # không đủ dữ liệu, bỏ qua âm thầm (không phải lỗi nghiêm trọng)

            vung_nen = xac_dinh_vung_nen(df, lookback_vung_nen, min_ngay_vung_nen)
            if vung_nen is None:
                continue  # mã không có giai đoạn tích lũy rõ ràng -> không áp dụng được bộ lọc này

            muc_giam = tinh_muc_giam_tu_diem_gay(df, vung_nen)
            rsi_volume = kiem_tra_rsi_va_volume(df, nguong_rsi, nguong_volume_ratio)

            dat_tieu_chi_1 = muc_giam["da_dut_gay"] and muc_giam["pct_giam_tu_pivot"] >= nguong_giam_toi_thieu_pct
            dat_tieu_chi_2 = rsi_volume["dat_rsi"]
            dat_tieu_chi_3 = rsi_volume["dat_volume"]

            if dat_tieu_chi_1 and dat_tieu_chi_2 and dat_tieu_chi_3:
                rows.append({
                    "ma": ma,
                    "gia_pivot_ho_tro": vung_nen["gia_pivot_ho_tro"],
                    "so_phien_vung_nen": vung_nen["so_phien_tich_luy"],
                    "gia_hien_tai": muc_giam["gia_hien_tai"],
                    "pct_giam_tu_pivot": muc_giam["pct_giam_tu_pivot"],
                    "rsi_hien_tai": rsi_volume["rsi_hien_tai"],
                    "volume_ratio": rsi_volume["volume_ratio"],
                })
        except Exception:  # noqa: BLE001 — cố ý bắt rộng, không dừng cả vòng quét
            continue  # ghi log lỗi ở tầng gọi thực tế nếu cần, không in ra đây

    df_ket_qua = pd.DataFrame(rows)
    if not df_ket_qua.empty:
        df_ket_qua = df_ket_qua.sort_values("pct_giam_tu_pivot", ascending=False).reset_index(drop=True)
    return df_ket_qua
