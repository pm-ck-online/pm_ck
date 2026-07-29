"""
core/character_integration.py
================================
Lớp TÍCH HỢP giữa `core/stock_character_classifier.py` (phân loại tính
cách giao dịch — DỨT KHOÁT/BÙNG NỔ/LÌNH XÌNH) và 2 module đã có:
    - `core/stock_signal_engine.py` (tín hiệu Mua/Bán)
    - `core/capital_allocator.py` (khuyến nghị phân bổ vốn)

Nguyên tắc: module này KHÔNG viết lại logic gốc của 2 module trên — chỉ
ĐIỀU CHỈNH kết quả đã có (chiết khấu độ tin cậy, giảm tỷ trọng phân bổ)
dựa trên tính cách giao dịch của mã, và cung cấp hàm quét hàng loạt tính
cách cho toàn watchlist.

CHỈ ĐỌC DỮ LIỆU / ĐIỀU CHỈNH KẾT QUẢ ĐÃ CÓ — KHÔNG đặt lệnh, KHÔNG khuyến
nghị đầu tư cá nhân hóa.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import pandas as pd

from core.stock_character_classifier import (
    NHAN_BUNG_NO_NGAN,
    gioi_han_ty_trong_theo_tinh_cach,
    he_so_chiet_khau_do_tin_cay,
    phan_loai_tinh_cach_co_phieu,
)


def dieu_chinh_tin_hieu_theo_tinh_cach(tin_hieu: dict, tinh_cach: dict) -> dict:
    """Điều chỉnh kết quả `evaluate_stock_signal()` (stock_signal_engine.py)
    dựa trên tính cách giao dịch của mã:
        - CHIẾT KHẤU `stock_score` nếu tín hiệu MUA theo mẫu hình BREAKOUT
          VÀ mã đang có `choppiness_score` cao (>1.0 — dễ là breakout giả,
          xem `he_so_chiet_khau_do_tin_cay()`).
        - GỘP THÊM cảnh báo SQUAT/CHURNING (nếu có) từ kết quả tính cách
          vào danh sách `canh_bao` của tín hiệu.
        - Gắn thêm `nhan_tinh_cach` vào output để tham khảo.

    KHÔNG sửa `tin_hieu` gốc — trả về dict MỚI.
    """
    kq = copy.deepcopy(tin_hieu)
    kq["nhan_tinh_cach"] = tinh_cach.get("nhan_tinh_cach")

    canh_bao_moi = list(kq.get("canh_bao") or [])
    for cb in tinh_cach.get("canh_bao", []):
        if cb not in canh_bao_moi:
            canh_bao_moi.append(cb)

    la_mua_breakout = (
        kq.get("khuyen_nghi") == "MUA"
        and (kq.get("chi_tiet") or {}).get("mau_hinh_ky_thuat") == "BREAKOUT"
    )

    if la_mua_breakout and kq.get("stock_score") is not None:
        he_so = he_so_chiet_khau_do_tin_cay(tinh_cach.get("choppiness_score", 0.0))
        if he_so < 1.0:
            kq["stock_score"] = kq["stock_score"] * he_so
            canh_bao_moi.append(
                f"Đã chiết khấu độ tin cậy tín hiệu Breakout do choppiness_score "
                f"cao (dễ là breakout giả) — hệ số x{he_so}."
            )

    kq["canh_bao"] = canh_bao_moi
    return kq


def dieu_chinh_phan_bo_theo_tinh_cach(alloc: dict, tinh_cach: dict) -> dict:
    """Điều chỉnh kết quả `get_allocation_recommendation()` (capital_allocator.py)
    dựa trên tính cách giao dịch của mã — dùng lại NGUYÊN
    `gioi_han_ty_trong_theo_tinh_cach()` đã có trong `stock_character_classifier.py`:
        - GIẢM `target_pct` (và `max_position_size` theo đúng tỷ lệ) nếu mã
          có nhãn BÙNG_NỔ_NGẮN hoặc đang có cảnh báo CHURNING.

    KHÔNG sửa `alloc` gốc — trả về dict MỚI.
    """
    kq = copy.deepcopy(alloc)
    nhan = tinh_cach.get("nhan_tinh_cach")
    canh_bao = tinh_cach.get("canh_bao", [])

    target_goc = kq.get("target_pct", 0.0) or 0.0
    target_moi = gioi_han_ty_trong_theo_tinh_cach(nhan, canh_bao, target_goc)

    if target_moi < target_goc:
        kq["target_pct"] = target_moi

        if kq.get("max_position_size") is not None and target_goc:
            ty_le = target_moi / target_goc
            kq["max_position_size"] = kq["max_position_size"] * ty_le

        ly_do = []
        if nhan == NHAN_BUNG_NO_NGAN:
            ly_do.append("nhãn BÙNG_NỔ_NGẮN")
        if any("CHURNING" in c for c in canh_bao):
            ly_do.append("cảnh báo CHURNING")

        kq["notes"] = list(kq.get("notes") or []) + [
            f"Đã giảm tỷ trọng phân bổ do {', '.join(ly_do)} — ưu tiên bảo toàn vốn."
        ]

    return kq


def quet_tinh_cach_watchlist(
    watchlist: list[str],
    lay_ohlcv_fn: Callable[[str], pd.DataFrame],
) -> pd.DataFrame:
    """Quét tính cách giao dịch cho TOÀN BỘ watchlist — gọi
    `phan_loai_tinh_cach_co_phieu()` cho từng mã. Nếu MỘT mã gặp lỗi (dữ
    liệu không đủ, exception bất kỳ khi lấy OHLCV...), KHÔNG dừng cả
    lượt quét — ghi nhận lỗi vào cột `loi` cho riêng mã đó, các mã còn
    lại vẫn được xử lý và trả về đầy đủ.

    `lay_ohlcv_fn(ma) -> pd.DataFrame`: hàm lấy dữ liệu OHLCV cho 1 mã
    (người gọi tự cung cấp, thường là `collector.get_ohlcv` hoặc đọc từ
    storage đã lưu sẵn).

    Trả về DataFrame với cột `ma`, `loi` (None nếu thành công), cùng toàn
    bộ các trường kết quả từ `phan_loai_tinh_cach_co_phieu()`.
    """
    rows: list[dict] = []
    for ma in watchlist:
        try:
            df = lay_ohlcv_fn(ma)
            ket_qua = phan_loai_tinh_cach_co_phieu(ma, df)
            row = {"ma": ma, "loi": None}
            row.update({k: v for k, v in ket_qua.items() if k != "ma"})
        except Exception as exc:  # noqa: BLE001
            row = {"ma": ma, "loi": f"{type(exc).__name__}: {exc}", "nhan_tinh_cach": None}
        rows.append(row)

    return pd.DataFrame(rows)
