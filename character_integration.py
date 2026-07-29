"""
core/character_integration.py

Các hàm "keo dán" (glue) kết hợp output của:
  - core.stock_character_classifier.phan_loai_tinh_cach_co_phieu()
  - core.stock_signal_engine.evaluate_stock_signal()
  - core.capital_allocator.get_allocation_recommendation()

THIẾT KẾ KHÔNG XÂM LẤN: không sửa đổi 3 module trên (stock_signal_engine.py,
capital_allocator.py, data_collector.py) — chỉ nhận output CÓ SẴN của chúng
và trả về bản đã điều chỉnh. Mục đích: thêm tính năng "tính cách cổ phiếu"
vào luồng mà không có rủi ro phá vỡ test suite hiện có (546 test đang pass
tính đến 27/7/2026).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from core.stock_character_classifier import (
    phan_loai_tinh_cach_co_phieu,
    he_so_chiet_khau_do_tin_cay,
    NHAN_BUNG_NO_NGAN,
    InsufficientDataError,
)


# ==============================================================================
# 1. Điều chỉnh output của evaluate_stock_signal() (core/stock_signal_engine.py)
# ==============================================================================

def dieu_chinh_tin_hieu_theo_tinh_cach(ket_qua_tin_hieu: dict, ket_qua_tinh_cach: dict) -> dict:
    """
    Nhận output CỦA CHÍNH evaluate_stock_signal() và
    phan_loai_tinh_cach_co_phieu(), trả về BẢN SAO đã điều chỉnh:

      - Nếu khuyến nghị là "MUA" theo mẫu hình "BREAKOUT" (khớp đúng khóa
        chi_tiet.mau_hinh_ky_thuat của stock_signal_engine.py) và
        choppiness_score đang cao -> chiết khấu stock_score, vì tín hiệu
        breakout trên nền mã có "tính cách lình xình" lịch sử dễ là
        breakout giả hơn (đúng Mục 8.1 của prompt gốc).
      - Gộp thêm cảnh báo SQUAT/CHURNING (nếu có) vào canh_bao.

    KHÔNG sửa `ket_qua_tin_hieu` gốc — trả về dict mới (shallow-copy an toàn
    vì các giá trị con không bị sửa tại chỗ, chỉ bị thay thế toàn bộ).
    """
    kq = dict(ket_qua_tin_hieu)
    kq["chi_tiet"] = dict(kq.get("chi_tiet") or {})
    kq["canh_bao"] = list(kq.get("canh_bao") or []) + list(ket_qua_tinh_cach.get("canh_bao") or [])

    mau_hinh = kq["chi_tiet"].get("mau_hinh_ky_thuat")
    if kq.get("khuyen_nghi") == "MUA" and mau_hinh == "BREAKOUT" and kq.get("stock_score") is not None:
        he_so = he_so_chiet_khau_do_tin_cay(ket_qua_tinh_cach.get("choppiness_score", 0.0))
        if he_so < 1.0:
            kq["stock_score"] = round(kq["stock_score"] * he_so, 3)
            kq["canh_bao"].append(
                f"Độ tin cậy tín hiệu Breakout đã chiết khấu x{he_so} do mã có tính cách "
                f"LÌNH_XÌNH lịch sử (choppiness_score="
                f"{ket_qua_tinh_cach.get('choppiness_score')})"
            )

    kq["nhan_tinh_cach"] = ket_qua_tinh_cach.get("nhan_tinh_cach")
    kq["character_score"] = ket_qua_tinh_cach.get("character_score")
    return kq


# ==============================================================================
# 2. Điều chỉnh output của get_allocation_recommendation() (core/capital_allocator.py)
# ==============================================================================

def dieu_chinh_phan_bo_theo_tinh_cach(allocation_result: dict, ket_qua_tinh_cach: dict) -> dict:
    """
    Nhận output CỦA CHÍNH get_allocation_recommendation() (có khóa
    target_pct, max_position_size, notes...) và phan_loai_tinh_cach_co_phieu(),
    trả về BẢN SAO đã điều chỉnh:

      - Nếu nhãn tính cách là BUNG_NO_NGAN hoặc có cờ cảnh báo CHURNING
        -> giảm target_pct và max_position_size còn 50%, thêm ghi chú giải
        thích lý do vào "notes" (đúng Mục 8.3 của prompt gốc).

    KHÔNG sửa `allocation_result` gốc — trả về dict mới.
    """
    kq = dict(allocation_result)
    kq["notes"] = list(kq.get("notes") or [])

    nhan = ket_qua_tinh_cach.get("nhan_tinh_cach")
    canh_bao_tinh_cach = ket_qua_tinh_cach.get("canh_bao") or []
    can_giam = nhan == NHAN_BUNG_NO_NGAN or any("CHURNING" in c for c in canh_bao_tinh_cach)

    if can_giam:
        if kq.get("target_pct") is not None:
            kq["target_pct"] = round(kq["target_pct"] * 0.5, 4)
        if kq.get("max_position_size") is not None:
            kq["max_position_size"] = int(kq["max_position_size"] * 0.5)
        ly_do = "tính cách BÙNG_NỔ_NGẮN" if nhan == NHAN_BUNG_NO_NGAN else "cảnh báo CHURNING"
        kq["notes"].append(
            f"Đã giảm 50% tỷ trọng/khối lượng khuyến nghị do {ly_do} "
            f"(xem core/stock_character_classifier.py)."
        )

    kq["nhan_tinh_cach"] = nhan
    return kq


# ==============================================================================
# 3. Quét toàn bộ watchlist — trả về bảng tổng hợp tính cách từng mã
# ==============================================================================

def quet_tinh_cach_watchlist(
    danh_sach_ma: list[str],
    lay_ohlcv_fn,
    lookback_window: int = 20,
    history_window: int = 500,
) -> pd.DataFrame:
    """
    Chạy phan_loai_tinh_cach_co_phieu() cho TOÀN BỘ watchlist, KHÔNG dừng
    cả vòng lặp nếu 1 mã lỗi/thiếu dữ liệu (chỉ ghi nhận lỗi cho riêng mã đó).

    Parameters
    ----------
    danh_sach_ma : list mã cổ phiếu, ví dụ ["HPG", "SSI", "VIB", ...]
    lay_ohlcv_fn : callable(symbol: str) -> pd.DataFrame — truyền vào hàm
        lấy dữ liệu đã có sẵn trong dự án, ví dụ:
            collector = DataCollector(source=..., config=...)
            quet_tinh_cach_watchlist(watchlist, lambda ma: collector.get_ohlcv(ma))
    lookback_window, history_window : xem phan_loai_tinh_cach_co_phieu()

    Returns
    -------
    pd.DataFrame — mỗi dòng 1 mã, đủ cột để lọc/sắp xếp nhanh trên dashboard
    (ví dụ: streamlit) hoặc xuất CSV. Mã lỗi vẫn có 1 dòng với cột "loi"
    ghi rõ nguyên nhân, các cột chỉ số khác để NaN.
    """
    rows = []
    for ma in danh_sach_ma:
        try:
            df = lay_ohlcv_fn(ma)
            if df is None or df.empty:
                rows.append({"ma": ma, "loi": "Không lấy được dữ liệu OHLCV (rỗng)."})
                continue
            ket_qua = phan_loai_tinh_cach_co_phieu(
                ma, df, lookback_window=lookback_window, history_window=history_window
            )
            rows.append(
                {
                    "ma": ket_qua["ma"],
                    "nhan_tinh_cach": ket_qua["nhan_tinh_cach"],
                    "character_score": ket_qua["character_score"],
                    "choppiness_score": ket_qua["choppiness_score"],
                    "streak_hien_tai": ket_qua["chi_tiet"]["streak_hien_tai"],
                    "velocity_10_phien_pct": ket_qua["chi_tiet"]["velocity_10_phien_pct"],
                    "choppiness_index": ket_qua["chi_tiet"]["choppiness_index"],
                    "so_canh_bao": len(ket_qua["canh_bao"]),
                    "canh_bao": "; ".join(ket_qua["canh_bao"]) if ket_qua["canh_bao"] else "",
                    "khuyen_nghi_chien_luoc": ket_qua["khuyen_nghi_chien_luoc"],
                    "do_tin_cay_thap": ket_qua["do_tin_cay_thap"],
                    "loi": None,
                }
            )
        except InsufficientDataError as e:
            rows.append({"ma": ma, "loi": f"Thiếu dữ liệu: {e}"})
        except Exception as e:  # noqa: BLE001 — cố ý bắt rộng để không dừng cả vòng lặp quét
            rows.append({"ma": ma, "loi": f"Lỗi không xác định: {type(e).__name__}: {e}"})

    return pd.DataFrame(rows)
