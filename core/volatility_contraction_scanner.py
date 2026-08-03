"""
volatility_contraction_scanner.py
====================================
Rà soát mô hình CO HẸP BIÊN ĐỘ DAO ĐỘNG (Volatility Contraction Pattern —
VCP) cho XAUUSD (vàng thế giới) và BTC/USD (Bitcoin) — chuỗi các chu kỳ
đỉnh-đáy liên tiếp có biên độ % GIẢM DẦN ĐỀU ĐẶN theo thời gian (ví dụ:
>20% → ~15% → ~10% → ~5%) trong một khoảng thời gian gần đây.

Ý NGHĨA: biên độ dao động thu hẹp dần thường là dấu hiệu thị trường đang
"nén" trước khi bứt phá mạnh theo 1 hướng — cùng logic đã áp dụng trong
`core/pattern_detector.py` (`detect_narrowing_pattern`) cho cổ phiếu VN,
nay mở rộng sang 2 tài sản quốc tế qua Binance Public API (xem
`core.data_collector.BinanceDataSource`).

CHỈ SÀNG LỌC/RÀ SOÁT MẪU HÌNH KỸ THUẬT THAM KHẢO — KHÔNG phải tín hiệu
giao dịch hay khuyến nghị đầu tư. Vàng và Bitcoin đều là tài sản biến
động mạnh, mô hình co hẹp KHÔNG đảm bảo hướng breakout sẽ xảy ra theo
chiều nào.
"""

from __future__ import annotations

import datetime as _dt
from typing import Callable, Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}

# Ngưỡng phân bậc biên độ mặc định — khớp đúng ví dụ trong tài liệu gốc
# (>20% → ~15% → ~10% → ~5%). Có thể truyền ngưỡng khác cho từng symbol
# (yêu cầu mục 10.3 tài liệu gốc: BTC/USD biến động mạnh hơn XAUUSD rất
# nhiều, nên cân nhắc bộ ngưỡng riêng cho từng symbol).
NGUONG_BAC_BIEN_DO_MAC_DINH = [20.0, 15.0, 10.0, 5.0, 3.0]


class InvalidVolatilityContractionError(ValueError):
    """Dữ liệu đầu vào không hợp lệ cho module rà soát mô hình co hẹp."""


def _validate_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise InvalidVolatilityContractionError("df_ohlc rỗng hoặc None.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise InvalidVolatilityContractionError(f"df_ohlc thiếu cột bắt buộc: {missing}.")


# ==============================================================================
# BƯỚC 1 — XÁC ĐỊNH ĐỈNH/ĐÁY CỤC BỘ (Local Peaks & Troughs)
# ==============================================================================

def tim_dinh_day_cuc_bo(df: pd.DataFrame, khoang_cach_toi_thieu: int = 3) -> list[dict]:
    """Dùng thuật toán swing high/low (đỉnh/đáy cục bộ) — 1 điểm được coi
    là ĐỈNH nếu giá cao hơn `khoang_cach_toi_thieu` phiên liền trước VÀ
    liền sau; tương tự cho ĐÁY. Trả về danh sách các điểm XEN KẼ đỉnh-đáy
    theo đúng thứ tự thời gian (đã lọc bỏ các đỉnh/đáy liên tiếp cùng
    loại, chỉ giữ điểm cực trị nhất giữa 2 lần đổi chiều).
    """
    from scipy.signal import argrelextrema

    _validate_df(df)
    df = df.reset_index(drop=True)

    idx_dinh = argrelextrema(df["high"].to_numpy(), np.greater_equal, order=khoang_cach_toi_thieu)[0]
    idx_day = argrelextrema(df["low"].to_numpy(), np.less_equal, order=khoang_cach_toi_thieu)[0]

    diem = [{"idx": int(i), "loai": "dinh", "gia": float(df["high"].iloc[i]), "date": df["date"].iloc[i]} for i in idx_dinh]
    diem += [{"idx": int(i), "loai": "day", "gia": float(df["low"].iloc[i]), "date": df["date"].iloc[i]} for i in idx_day]
    diem.sort(key=lambda d: d["idx"])

    # Lọc để đảm bảo XEN KẼ đỉnh-đáy (loại bỏ 2 đỉnh/2 đáy liên tiếp, chỉ
    # giữ điểm cực trị hơn giữa 2 lần đổi chiều).
    diem_da_loc: list[dict] = []
    for d in diem:
        if diem_da_loc and diem_da_loc[-1]["loai"] == d["loai"]:
            if (d["loai"] == "dinh" and d["gia"] > diem_da_loc[-1]["gia"]) or \
               (d["loai"] == "day" and d["gia"] < diem_da_loc[-1]["gia"]):
                diem_da_loc[-1] = d  # thay bằng điểm cực trị hơn
        else:
            diem_da_loc.append(d)
    return diem_da_loc


# ==============================================================================
# BƯỚC 2 — TÍNH BIÊN ĐỘ % TỪNG CHU KỲ ĐỈNH-ĐÁY
# ==============================================================================

def tinh_bien_do_tung_chu_ky(diem_dinh_day: list[dict]) -> list[dict]:
    """Mỗi "chu kỳ" = 1 cặp (đỉnh, đáy liền sau) HOẶC (đáy, đỉnh liền sau)
    — tính % biên độ dao động = |giá_sau - giá_trước| / giá_trước × 100.
    Trả về danh sách chu kỳ theo thứ tự thời gian, kèm % biên độ.
    """
    chu_ky = []
    for i in range(len(diem_dinh_day) - 1):
        a, b = diem_dinh_day[i], diem_dinh_day[i + 1]
        if a["gia"] == 0:
            continue
        bien_do_pct = abs(b["gia"] - a["gia"]) / a["gia"] * 100
        chu_ky.append({
            "tu_ngay": a["date"], "den_ngay": b["date"],
            "tu_loai": a["loai"], "den_loai": b["loai"],
            "bien_do_pct": round(bien_do_pct, 2),
        })
    return chu_ky


# ==============================================================================
# BƯỚC 3 — XÁC NHẬN CHUỖI CO HẸP DẦN (Progressive Contraction Check)
# ==============================================================================

def xac_nhan_chuoi_co_hep(
    chu_ky: list[dict], so_chu_ky_toi_thieu: int = 3, dung_sai_pct: float = 3.0
) -> dict:
    """Kiểm tra xem `so_chu_ky_toi_thieu` chu kỳ GẦN NHẤT có biên độ %
    giảm dần đều hay không (mỗi chu kỳ sau <= chu kỳ trước + dung_sai_pct,
    cho phép sai số nhỏ vì thị trường hiếm khi giảm biên độ "đẹp" tuyệt
    đối).
    """
    if len(chu_ky) < so_chu_ky_toi_thieu:
        return {
            "hop_le": False,
            "ly_do": f"Chỉ có {len(chu_ky)} chu kỳ, cần tối thiểu {so_chu_ky_toi_thieu}.",
            "so_chu_ky_da_xet": len(chu_ky),
        }

    chuoi_gan_nhat = chu_ky[-so_chu_ky_toi_thieu:]
    bien_do_list = [c["bien_do_pct"] for c in chuoi_gan_nhat]

    hop_le = all(
        bien_do_list[i + 1] <= bien_do_list[i] + dung_sai_pct
        for i in range(len(bien_do_list) - 1)
    )

    return {
        "hop_le": hop_le,
        "chuoi_bien_do": bien_do_list,
        "ty_le_giam_tong_the": round(
            (bien_do_list[0] - bien_do_list[-1]) / bien_do_list[0] * 100, 1
        ) if bien_do_list[0] > 0 else None,
        "so_chu_ky_da_xet": len(chuoi_gan_nhat),
    }


def gan_nhan_bac_bien_do(bien_do_pct: float, nguong_bac: Optional[list[float]] = None) -> str:
    """Gán nhãn bậc biên độ gần nhất (hiển thị trực quan dạng '>20%', '~15%'...).

    `nguong_bac`: cho phép truyền bộ ngưỡng RIÊNG theo từng symbol (mục
    10.3 tài liệu gốc — BTC/USD biến động mạnh hơn XAUUSD rất nhiều, nên
    dùng bộ ngưỡng khác nhau thay vì dùng chung 1 bộ cho cả 2).
    """
    nguong = nguong_bac if nguong_bac is not None else NGUONG_BAC_BIEN_DO_MAC_DINH
    for i, n in enumerate(nguong):
        if bien_do_pct >= n:
            return f">{n:.0f}%" if i == 0 else f"~{n:.0f}%"
    return f"<{nguong[-1]:.0f}%"


# ==============================================================================
# BƯỚC 4 — ĐỐI CHIẾU VỚI ĐƯỜNG TB 20 NGÀY (MA20)
# ==============================================================================

def doi_chieu_voi_ma20(df: pd.DataFrame) -> dict:
    """Trả về vị trí giá hiện tại so với MA20 (trên/dưới, % khoảng cách)
    và xu hướng MA20 (đang tăng/giảm/đi ngang, dựa trên độ dốc 5 phiên
    gần nhất) — dùng làm lớp xác nhận bổ sung, KHÔNG dùng để tự quyết
    định, chỉ tăng/giảm độ tin cậy của tín hiệu co hẹp.
    """
    _validate_df(df)
    ma20 = df["close"].rolling(20).mean()
    gia_hien_tai = float(df["close"].iloc[-1])
    ma20_hien_tai = ma20.iloc[-1]

    if pd.isna(ma20_hien_tai) or ma20_hien_tai == 0:
        return {"gia_tren_ma20": None, "khoang_cach_pct": None, "ma20_xu_huong": None}

    khoang_cach_pct = (gia_hien_tai - ma20_hien_tai) / ma20_hien_tai * 100
    do_doc_ma20 = (
        (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5] * 100
        if len(ma20) >= 5 and pd.notna(ma20.iloc[-5]) and ma20.iloc[-5] != 0 else None
    )

    return {
        "gia_tren_ma20": gia_hien_tai > ma20_hien_tai,
        "khoang_cach_pct": round(khoang_cach_pct, 2),
        "ma20_xu_huong": (
            "TANG" if do_doc_ma20 is not None and do_doc_ma20 > 0.3 else
            "GIAM" if do_doc_ma20 is not None and do_doc_ma20 < -0.3 else
            "DI_NGANG"
        ),
    }


# ==============================================================================
# HỆ SỐ NẾN THEO KHUNG THỜI GIAN (Binance)
# ==============================================================================

def _he_so_nen_theo_khung(khung: str) -> float:
    """Số "nến" xấp xỉ tương ứng 1 ngày cho từng khung Binance, để cắt
    đúng độ dài cửa sổ theo ngày thực (Binance dùng ký hiệu chữ thường).
    """
    return {"1d": 1, "4h": 6, "1w": 1 / 7, "1h": 24}.get(khung, 1)


# ==============================================================================
# TỰ ĐỘNG CHỌN KHUNG THỜI GIAN
# ==============================================================================

def tu_chon_khung_thoi_gian(
    symbol: str,
    lay_ohlcv_fn: Callable[..., pd.DataFrame],
    khung_ung_vien: list[str],
    so_ngay_tham_chieu: int,
    so_chu_ky_toi_thieu: int,
    dung_sai_pct: float = 3.0,
) -> dict:
    """Thử LẦN LƯỢT từng khung thời gian trong `khung_ung_vien` (đúng cú
    pháp Binance, chữ thường: "1d", "4h", "1w"...), chạy toàn bộ pipeline
    (tìm đỉnh/đáy -> tính biên độ -> xác nhận co hẹp) cho mỗi khung, CHỌN
    khung đầu tiên cho kết quả `hop_le=True` với `ty_le_giam_tong_the`
    lớn nhất (co hẹp rõ rệt nhất). Nếu không khung nào hợp lệ, trả về kết
    quả của khung có nhiều chu kỳ nhất (gần hợp lệ nhất) kèm cờ cảnh báo.
    """
    ket_qua_theo_khung: dict[str, dict] = {}
    for khung in khung_ung_vien:
        df = lay_ohlcv_fn(symbol, timeframe=khung)
        so_nen = max(1, int(so_ngay_tham_chieu * _he_so_nen_theo_khung(khung)))
        df_scope = df.tail(so_nen)

        diem = tim_dinh_day_cuc_bo(df_scope)
        chu_ky = tinh_bien_do_tung_chu_ky(diem)
        xac_nhan = xac_nhan_chuoi_co_hep(chu_ky, so_chu_ky_toi_thieu, dung_sai_pct)
        ket_qua_theo_khung[khung] = {**xac_nhan, "chu_ky_chi_tiet": chu_ky}

    hop_le = {k: v for k, v in ket_qua_theo_khung.items() if v.get("hop_le")}
    if hop_le:
        khung_tot_nhat = max(hop_le, key=lambda k: hop_le[k].get("ty_le_giam_tong_the") or 0)
        return {
            "khung_da_chon": khung_tot_nhat,
            **ket_qua_theo_khung[khung_tot_nhat],
            "tat_ca_khung": ket_qua_theo_khung,
        }

    khung_gan_nhat = max(ket_qua_theo_khung, key=lambda k: ket_qua_theo_khung[k].get("so_chu_ky_da_xet", 0))
    return {
        "khung_da_chon": khung_gan_nhat,
        **ket_qua_theo_khung[khung_gan_nhat],
        "canh_bao": "Không có khung thời gian nào xác nhận chuỗi co hẹp rõ ràng.",
        "tat_ca_khung": ket_qua_theo_khung,
    }


# ==============================================================================
# HÀM CHÍNH — GHÉP TOÀN BỘ PIPELINE
# ==============================================================================

def rao_soat_mo_hinh_co_hep(
    symbol: str,
    lay_ohlcv_fn: Callable[..., pd.DataFrame],
    khung_thoi_gian_ung_vien: tuple[str, ...] = ("1d", "4h"),
    so_ngay_tham_chieu: int = 45,
    so_chu_ky_toi_thieu: int = 3,
    dung_sai_pct: float = 3.0,
    nguong_bac_bien_do: Optional[list[float]] = None,
) -> dict:
    """Hàm chính: chạy toàn bộ rà soát cho 1 symbol ("XAUUSD" hoặc
    "BTCUSD", tự động ánh xạ sang PAXGUSDT/BTCUSDT khi gọi Binance qua
    `lay_ohlcv_fn` — xem `core.data_collector.BinanceDataSource`).

    `nguong_bac_bien_do`: bộ ngưỡng phân bậc RIÊNG cho symbol này (nên
    dùng ngưỡng khác nhau cho XAUUSD và BTCUSD vì BTC biến động mạnh hơn
    nhiều — mục 10.3 tài liệu gốc). Để None dùng mặc định chung.

    CHỈ RÀ SOÁT MẪU HÌNH KỸ THUẬT THAM KHẢO — KHÔNG phải tín hiệu giao
    dịch hay khuyến nghị đầu tư.
    """
    ket_qua_khung = tu_chon_khung_thoi_gian(
        symbol, lay_ohlcv_fn, list(khung_thoi_gian_ung_vien),
        so_ngay_tham_chieu, so_chu_ky_toi_thieu, dung_sai_pct,
    )

    df_khung_da_chon = lay_ohlcv_fn(symbol, timeframe=ket_qua_khung["khung_da_chon"])
    ma20_info = doi_chieu_voi_ma20(df_khung_da_chon)

    chuoi_bac = [
        gan_nhan_bac_bien_do(b, nguong_bac_bien_do)
        for b in ket_qua_khung.get("chuoi_bien_do", [])
    ]

    return {
        "symbol": symbol,
        "khung_thoi_gian_da_chon": ket_qua_khung["khung_da_chon"],
        "xac_nhan_co_hep": ket_qua_khung["hop_le"],
        "chuoi_bien_do_pct": ket_qua_khung.get("chuoi_bien_do"),
        "chuoi_bac_bien_do": chuoi_bac,
        "ty_le_giam_tong_the_pct": ket_qua_khung.get("ty_le_giam_tong_the"),
        "doi_chieu_ma20": ma20_info,
        "canh_bao": ket_qua_khung.get("canh_bao"),
        "chi_tiet_chu_ky": ket_qua_khung.get("chu_ky_chi_tiet"),
        "ngay_danh_gia": _dt.date.today().isoformat(),
        "canh_bao_phap_ly": (
            "Đây là công cụ rà soát mẫu hình kỹ thuật tham khảo, KHÔNG phải "
            "tín hiệu giao dịch hay khuyến nghị đầu tư. Vàng và Bitcoin đều "
            "là tài sản biến động mạnh, mô hình co hẹp không đảm bảo hướng "
            "breakout sẽ xảy ra theo chiều nào."
        ),
    }
