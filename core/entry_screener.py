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

def xep_hang_uu_tien_theo_duong_tham_chieu(
    gia_dong_cua: float, duong_tham_chieu: Optional[float], ten_duong: str = "EMA200"
) -> dict:
    """Phiên bản TỔNG QUÁT của `xep_hang_uu_tien_theo_ema200()` — dùng
    chung được cho BẤT KỲ đường trung bình nào làm mốc tham chiếu (EMA200
    hoặc MA20), theo đúng cùng 1 công thức/ngưỡng (bổ sung 04/08/2026 để
    hỗ trợ lựa chọn MA20 làm đường tham chiếu thay thế EMA200).

    `ten_duong`: chỉ dùng để đặt tên trong thông báo lỗi cho khớp ngữ
    cảnh hiển thị (VD: "EMA200" hoặc "MA20").
    """
    if duong_tham_chieu is None:
        return {"do_lech_pct": None, "xep_hang_uu_tien": "KHONG_DAT"}
    if duong_tham_chieu <= 0:
        raise InvalidEntryScreenerError(f"{ten_duong} phải > 0.")

    do_lech = (gia_dong_cua - duong_tham_chieu) / duong_tham_chieu * 100
    if do_lech >= 0:
        hang = "UU_TIEN_CAO"
    elif do_lech >= -10.0:
        hang = "UU_TIEN_TRUNG_BINH"
    else:
        hang = "KHONG_DAT"

    return {"do_lech_pct": do_lech, "xep_hang_uu_tien": hang}


def xep_hang_uu_tien_theo_ema200(gia_dong_cua: float, ema200: Optional[float]) -> dict:
    """Xếp hạng ưu tiên vào lệnh dựa trên độ lệch giá so với EMA200 (mục 1
    tài liệu): trên EMA200 -> ưu tiên CAO, trong -10%..0% -> ưu tiên TRUNG
    BÌNH (có thể đang pullback), dưới -10% -> KHÔNG ĐẠT.

    `ema200=None` (mã chưa đủ dữ liệu tính EMA200) -> trả về KHÔNG ĐẠT,
    không báo lỗi (an toàn, loại khỏi danh sách chờ).

    (Giữ nguyên hàm này để KHÔNG phá vỡ các lời gọi hiện có — nay chỉ là
    lớp bọc mỏng gọi `xep_hang_uu_tien_theo_duong_tham_chieu()` ở trên,
    đảm bảo kết quả giống hệt như trước.)
    """
    ket_qua = xep_hang_uu_tien_theo_duong_tham_chieu(gia_dong_cua, ema200, "EMA200")
    return {"do_lech_ema200_pct": ket_qua["do_lech_pct"], "xep_hang_uu_tien": ket_qua["xep_hang_uu_tien"]}


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


# ==============================================================================
# MỤC 7 (BỔ SUNG 03/08/2026) — THỐNG KÊ XÁC SUẤT TĂNG/GIẢM THEO BẬC %,
# DỰA TRÊN TÌNH HUỐNG TƯƠNG TỰ TRONG LỊCH SỬ CỦA CHÍNH MÃ ĐÓ
# ==============================================================================

# Các bậc % dùng để phân loại mức tăng/giảm sau `so_phien_du_bao` phiên.
BAC_TANG_GIAM = [
    ("giam_tren_15", -float("inf"), -15.0, "Giảm > 15%"),
    ("giam_10_15", -15.0, -10.0, "Giảm 10-15%"),
    ("giam_5_10", -10.0, -5.0, "Giảm 5-10%"),
    ("giam_0_5", -5.0, 0.0, "Giảm 0-5%"),
    ("tang_0_5", 0.0, 5.0, "Tăng 0-5%"),
    ("tang_5_10", 5.0, 10.0, "Tăng 5-10%"),
    ("tang_10_15", 10.0, 15.0, "Tăng 10-15%"),
    ("tang_tren_15", 15.0, float("inf"), "Tăng > 15%"),
]


def _xac_dinh_bac(pct_thay_doi: float) -> str:
    for key, lo, hi, _ in BAC_TANG_GIAM:
        if lo < pct_thay_doi <= hi or (lo == -float("inf") and pct_thay_doi <= hi):
            return key
    return BAC_TANG_GIAM[-1][0]  # fallback: giá trị cực lớn -> "tang_tren_15"


def _khop_giai_doan(ngay, chuoi_giai_doan, giai_doan_loc: Optional[str]) -> bool:
    """Kiểm tra ngày `ngay` có thuộc đúng `giai_doan_loc` (uptrend/downtrend/
    sideway) theo `chuoi_giai_doan` (pd.Series index=ngày, xem
    `core.market_regime_detector.tinh_chuoi_giai_doan_theo_ngay`) hay không.

    Nếu KHÔNG lọc theo giai đoạn (`giai_doan_loc` là None) -> luôn khớp
    (True), giữ nguyên hành vi cũ. Nếu ngày đó KHÔNG có trong chuỗi giai
    đoạn (thiếu dữ liệu) -> KHÔNG khớp (loại khỏi thống kê, không suy diễn).
    """
    if giai_doan_loc is None:
        return True
    if chuoi_giai_doan is None:
        return True
    gia_tri = chuoi_giai_doan.get(pd.Timestamp(ngay))
    return gia_tri == giai_doan_loc


def tinh_thong_ke_tang_giam_lich_su(
    df_ohlcv: pd.DataFrame,
    tieu_chi_dat: list[str],
    so_phien_du_bao: int = 30,
    so_phien_kiem_tra: int = 250,
    duong_tham_chieu: str = "ema200",
    chuoi_giai_doan: Optional[pd.Series] = None,
    giai_doan_loc: Optional[str] = None,
) -> dict:
    """Với CHÍNH mã đang xét, quét lại lịch sử giá để tìm các thời điểm
    TRONG QUÁ KHỨ mã này từng thỏa CÙNG bộ tiêu chí (`tieu_chi_dat`) như
    hiện tại — rồi thống kê % thay đổi giá sau đúng `so_phien_du_bao`
    phiên kể từ mỗi thời điểm đó, phân theo các bậc % (xem `BAC_TANG_GIAM`).

    `duong_tham_chieu`: "ema200" (mặc định) hoặc "ma20" — đường trung
    bình dùng làm mốc cho tiêu chí "dieu_kien_nen_ema200" khi phát lại
    lịch sử (bổ sung 04/08/2026, cho phép chọn MA20 thay EMA200).

    `chuoi_giai_doan` + `giai_doan_loc` (bổ sung 05/08/2026): nếu truyền
    cả 2, CHỈ tính các lần trong quá khứ mà NGÀY đó khớp đúng giai đoạn
    thị trường/ngành mong muốn ("uptrend"/"downtrend"/"sideway", xem
    `core.market_regime_detector.tinh_chuoi_giai_doan_theo_ngay`) — giúp
    trả lời "tỷ lệ thắng/thua trong quá khứ khi thị trường/ngành đang
    Uptrend có khác gì khi Downtrend không". Để trống (mặc định) -> giữ
    nguyên hành vi cũ, không lọc theo giai đoạn.

    PHẠM VI ĐÃ THU HẸP CÓ CHỦ Ý: chỉ phát lại 2 tiêu chí tính NHANH
    ("dieu_kien_nen_ema200", "tich_luy_dai_han") — 2 tiêu chí còn lại
    ("dao_dong_tat_dan", "volume_breakout") cần quét lại mô hình thu hẹp
    biên độ trên tới 30 tháng dữ liệu MỖI LẦN gọi
    (`pattern_detector.detect_narrowing_pattern`), quá chậm nếu phát lại
    hàng trăm lần trong 1 lượt tính cho dashboard — nên KHÔNG được phát
    lại ở đây. Nếu mã CHỈ đạt 2 tiêu chí này (không có tiêu chí nhanh
    nào), hàm trả về cỡ mẫu = 0 kèm ghi chú rõ ràng, KHÔNG suy diễn.

    Đây là TẦN SUẤT THỰC NGHIỆM từ lịch sử, KHÔNG phải xác suất dự báo
    tương lai được đảm bảo — luôn báo cáo kèm cỡ mẫu.
    """
    if duong_tham_chieu not in ("ema200", "ma20"):
        raise InvalidEntryScreenerError('duong_tham_chieu phải là "ema200" hoặc "ma20".')

    TIEU_CHI_NHANH = {"dieu_kien_nen_ema200", "tich_luy_dai_han"}
    tieu_chi_su_dung = [t for t in tieu_chi_dat if t in TIEU_CHI_NHANH]

    if not tieu_chi_su_dung:
        return {
            "so_lan_quan_sat": 0,
            "ghi_chu": (
                "Mã này chỉ đạt tiêu chí 'Mô hình thu hẹp biên độ' và/hoặc "
                "'Khối lượng breakout' — 2 tiêu chí này KHÔNG được phát lại "
                "trong thống kê lịch sử (quá chậm để tính hàng loạt), nên "
                "không có số liệu thống kê cho mã này."
            ),
            "phan_bo": {},
        }

    so_phien_toi_thieu = 200 if duong_tham_chieu == "ema200" else 20
    if df_ohlcv is None or df_ohlcv.empty or len(df_ohlcv) < so_phien_toi_thieu:
        return {"so_lan_quan_sat": 0, "ghi_chu": f"Chưa đủ dữ liệu lịch sử (cần tối thiểu {so_phien_toi_thieu} phiên).", "phan_bo": {}}

    from core.indicators import calculate_ema, calculate_ma

    n = len(df_ohlcv)
    if duong_tham_chieu == "ema200":
        duong_series = calculate_ema(df_ohlcv, period=200)
    else:
        duong_series = calculate_ma(df_ohlcv, period=20)
    closes = df_ohlcv["close"]

    start = so_phien_toi_thieu
    end = n - so_phien_du_bao  # cần đủ dữ liệu TƯƠNG LAI để đo % thay đổi
    start = min(start, max(0, n - so_phien_kiem_tra))

    ket_qua_tung_lan: list[float] = []
    for i in range(start, max(start, end)):
        duong_i = duong_series.iloc[i]
        if pd.isna(duong_i) or duong_i <= 0:
            continue
        close_i = float(closes.iloc[i])

        dat_dieu_kien = False
        if "dieu_kien_nen_ema200" in tieu_chi_su_dung:
            do_lech = (close_i - duong_i) / duong_i * 100
            if do_lech >= -10.0:
                dat_dieu_kien = True

        if not dat_dieu_kien and "tich_luy_dai_han" in tieu_chi_su_dung:
            tich_luy = kiem_tra_tich_luy_dai_han(df_ohlcv.iloc[: i + 1])
            if tich_luy.get("dat"):
                dat_dieu_kien = True

        if not dat_dieu_kien:
            continue

        if not _khop_giai_doan(df_ohlcv["date"].iloc[i], chuoi_giai_doan, giai_doan_loc):
            continue

        close_sau = float(closes.iloc[i + so_phien_du_bao])
        pct_thay_doi = (close_sau - close_i) / close_i * 100
        ket_qua_tung_lan.append(pct_thay_doi)

    so_lan = len(ket_qua_tung_lan)
    if so_lan == 0:
        return {
            "so_lan_quan_sat": 0,
            "ghi_chu": "Không tìm thấy tình huống tương tự nào trong lịch sử đã quét.",
            "phan_bo": {},
        }

    phan_bo = {}
    for key, _, _, nhan in BAC_TANG_GIAM:
        gia_tri_bac = [x for x in ket_qua_tung_lan if _xac_dinh_bac(x) == key]
        so_lan_bac = len(gia_tri_bac)
        phan_bo[key] = {
            "nhan": nhan,
            "so_lan": so_lan_bac,
            "ty_le_pct": round(so_lan_bac / so_lan * 100, 1),
            "gia_tri_trung_binh_pct": (
                round(sum(gia_tri_bac) / so_lan_bac, 2) if so_lan_bac > 0 else None
            ),
        }

    return {
        "so_lan_quan_sat": so_lan,
        "phan_bo": phan_bo,
        "pct_thay_doi_trung_binh": round(sum(ket_qua_tung_lan) / so_lan, 2),
        "ghi_chu": (
            f"Dựa trên {so_lan} lần trong quá khứ mã này thỏa tiêu chí: "
            + ", ".join(TIEU_CHI_KHA_DUNG.get(t, t) for t in tieu_chi_su_dung)
        ),
    }


# ==============================================================================
# MỤC 8 (BỔ SUNG 04/08/2026) — KELLY CRITERION DỰA TRÊN PHÂN BỐ TĂNG/GIẢM
# ==============================================================================

TANG_KEYS = ("tang_0_5", "tang_5_10", "tang_10_15", "tang_tren_15")
GIAM_KEYS = ("giam_0_5", "giam_5_10", "giam_10_15", "giam_tren_15")


def tinh_kelly_fraction(phan_bo: dict) -> dict:
    """Tính hệ số Kelly Criterion (f*) — tỷ trọng vốn TỐI ƯU (về mặt toán
    học, tối đa hóa tốc độ tăng trưởng vốn kỳ vọng dài hạn) cho 1 lệnh —
    dựa trên phân bố xác suất tăng/giảm theo bậc % đã tính ở
    `tinh_thong_ke_tang_giam_lich_su()`.

    CÔNG THỨC (Kelly cho cược nhị phân thắng tỷ lệ G / thua tỷ lệ L):
        f* = p/L - q/G
    trong đó:
        p = xác suất THẮNG = tổng ty_le_pct các bậc "tăng" / 100
        q = 1 - p = xác suất THUA
        G = trung bình % TĂNG khi thắng (bình quân gia quyền theo số lần
            quan sát của các bậc tăng), dạng phân số (VD 0,08 cho 8%)
        L = trung bình % GIẢM khi thua (lấy trị tuyệt đối), dạng phân số

    Trả về f* đã cắt về [0, 1] — f*=0 nghĩa là "không có lợi thế thống
    kê, không nên vào lệnh theo Kelly". Kèm f*/2 ("nửa Kelly") vì Kelly
    ĐẦY ĐỦ trên thực tế biến động rất mạnh, giới đầu tư định lượng
    thường khuyến nghị dùng 1/2 hoặc 1/4 Kelly để giảm rủi ro.

    LƯU Ý: đây là CÔNG THỨC TOÁN HỌC áp dụng lên DỮ LIỆU TẦN SUẤT LỊCH
    SỬ - không phải cam kết lợi nhuận, và giả định phân bố tương lai
    giống phân bố quá khứ (một giả định KHÔNG được đảm bảo).
    """
    so_lan_tang = sum(phan_bo.get(k, {}).get("so_lan", 0) for k in TANG_KEYS)
    so_lan_giam = sum(phan_bo.get(k, {}).get("so_lan", 0) for k in GIAM_KEYS)
    tong_so_lan = so_lan_tang + so_lan_giam

    if tong_so_lan == 0:
        return {
            "kelly_f": None, "kelly_f_nua": None,
            "ghi_chu": "Không có đủ số liệu (0 quan sát) để tính Kelly.",
        }

    p = so_lan_tang / tong_so_lan
    q = so_lan_giam / tong_so_lan

    if so_lan_tang == 0 or so_lan_giam == 0:
        return {
            "kelly_f": None, "kelly_f_nua": None,
            "xac_suat_thang": round(p * 100, 1), "xac_suat_thua": round(q * 100, 1),
            "ghi_chu": "Không thể tính Kelly vì thiếu quan sát ở chiều tăng hoặc giảm (toàn bộ lịch sử chỉ có 1 chiều).",
        }

    tong_gia_tri_tang = sum(
        phan_bo[k]["so_lan"] * phan_bo[k]["gia_tri_trung_binh_pct"]
        for k in TANG_KEYS if phan_bo.get(k, {}).get("so_lan", 0) > 0
    )
    tong_gia_tri_giam = sum(
        phan_bo[k]["so_lan"] * phan_bo[k]["gia_tri_trung_binh_pct"]
        for k in GIAM_KEYS if phan_bo.get(k, {}).get("so_lan", 0) > 0
    )
    G = (tong_gia_tri_tang / so_lan_tang) / 100
    L = abs(tong_gia_tri_giam / so_lan_giam) / 100

    if G <= 0 or L <= 0:
        return {
            "kelly_f": None, "kelly_f_nua": None,
            "xac_suat_thang": round(p * 100, 1), "xac_suat_thua": round(q * 100, 1),
            "ghi_chu": "Không thể tính Kelly do biên độ tăng/giảm trung bình bằng 0.",
        }

    f_sao = p / L - q / G
    f_sao_cat = max(0.0, min(1.0, f_sao))

    return {
        "kelly_f": round(f_sao_cat, 4),
        "kelly_f_nua": round(f_sao_cat / 2, 4),
        "kelly_f_tho": round(f_sao, 4),
        "xac_suat_thang": round(p * 100, 1),
        "xac_suat_thua": round(q * 100, 1),
        "trung_binh_tang_pct": round(G * 100, 2),
        "trung_binh_giam_pct": round(L * 100, 2),
        "ghi_chu": (
            "Không có lợi thế thống kê (Kelly <= 0) - về mặt toán học không nên vào lệnh."
            if f_sao <= 0 else
            f"Kelly đề xuất phân bổ tối đa {f_sao_cat * 100:.1f}% vốn cho lệnh này "
            f"(khuyến nghị thực tế nên dùng nửa Kelly {f_sao_cat / 2 * 100:.1f}% để giảm biến động)."
        ),
    }


# ==============================================================================
# MỤC 9 (BỔ SUNG 04/08/2026) — KIỂM ĐỊNH THỐNG KÊ SO SÁNH 2 KHOẢNG ĐỘ LỆCH
# ==============================================================================

def so_sanh_2_khoang_do_lech(
    df_ohlcv: pd.DataFrame,
    khoang_1: tuple[float, float],
    khoang_2: tuple[float, float],
    duong_tham_chieu: str = "ma20",
    so_phien_du_bao: int = 5,
    so_phien_kiem_tra: int = 250,
    nguong_y_nghia: float = 0.05,
    chuoi_giai_doan: Optional[pd.Series] = None,
    giai_doan_loc: Optional[str] = None,
) -> dict:
    """Kiểm định thống kê xem 2 KHOẢNG ĐỘ LỆCH (% giá so với đường tham
    chiếu EMA200/MA20) có khác biệt CÓ Ý NGHĨA THỐNG KÊ về tỷ lệ thắng
    hay không — trả lời đúng câu hỏi "khoảng -5% đến 0% có khác gì
    khoảng 0% đến +5% không", KHÔNG chỉ so sánh cảm tính 2 con số %.

    `chuoi_giai_doan` + `giai_doan_loc` (bổ sung 05/08/2026): nếu truyền
    cả 2, CHỈ tính các lần trong quá khứ mà NGÀY đó khớp đúng giai đoạn
    thị trường/ngành mong muốn (xem
    `core.market_regime_detector.tinh_chuoi_giai_doan_theo_ngay`).

    `khoang_1`, `khoang_2`: tuple (từ, đến) tính theo độ lệch % CÓ DẤU
    (bổ sung 04/08/2026 — trước đây dùng trị tuyệt đối, nay cho phép
    nhập số ÂM để phân biệt DƯỚI/TRÊN đường tham chiếu):
        - Số ÂM = giá đang DƯỚI đường tham chiếu (VD: -5 đến 0 nghĩa là
          giá thấp hơn đường tham chiếu tối đa 5%).
        - Số DƯƠNG = giá đang TRÊN đường tham chiếu (VD: 0 đến 5 nghĩa
          là giá cao hơn đường tham chiếu tối đa 5%).
    Mỗi khoảng là nửa khoảng đóng-mở: [từ, đến) — bao gồm cận dưới,
    không bao gồm cận trên.

    PHƯƠNG PHÁP: kiểm định 2 tỷ lệ (two-proportion z-test) so sánh tỷ lệ
    THẮNG (giá tăng sau `so_phien_du_bao` phiên) giữa 2 nhóm — trả về
    trị số z, p-value, và kết luận CÓ/KHÔNG khác biệt ý nghĩa thống kê ở
    mức `nguong_y_nghia` (mặc định 5%, tức p-value < 0,05 mới coi là có
    ý nghĩa — quy ước thống kê phổ biến).

    LƯU Ý: đây vẫn là kiểm định trên TẦN SUẤT LỊCH SỬ — p-value nhỏ chỉ
    cho biết khác biệt QUAN SÁT ĐƯỢC khó xảy ra do ngẫu nhiên thuần túy,
    KHÔNG chứng minh quan hệ nhân quả hay đảm bảo lặp lại trong tương lai.
    """
    from scipy.stats import norm

    if duong_tham_chieu not in ("ema200", "ma20"):
        raise InvalidEntryScreenerError('duong_tham_chieu phải là "ema200" hoặc "ma20".')
    if khoang_1[0] >= khoang_1[1]:
        raise InvalidEntryScreenerError('khoang_1: giá trị "từ" phải nhỏ hơn "đến".')
    if khoang_2[0] >= khoang_2[1]:
        raise InvalidEntryScreenerError('khoang_2: giá trị "từ" phải nhỏ hơn "đến".')

    so_phien_toi_thieu = 200 if duong_tham_chieu == "ema200" else 20
    if df_ohlcv is None or df_ohlcv.empty or len(df_ohlcv) < so_phien_toi_thieu:
        return {
            "hop_le": False,
            "ghi_chu": f"Chưa đủ dữ liệu lịch sử (cần tối thiểu {so_phien_toi_thieu} phiên).",
        }

    from core.indicators import calculate_ema, calculate_ma

    n = len(df_ohlcv)
    duong_series = calculate_ema(df_ohlcv, period=200) if duong_tham_chieu == "ema200" else calculate_ma(df_ohlcv, period=20)
    closes = df_ohlcv["close"]

    start = min(so_phien_toi_thieu, max(0, n - so_phien_kiem_tra))
    end = n - so_phien_du_bao

    ket_qua_theo_khoang = {1: [], 2: []}
    tong_so_phien_quet = 0
    for i in range(start, max(start, end)):
        duong_i = duong_series.iloc[i]
        if pd.isna(duong_i) or duong_i <= 0:
            continue
        if not _khop_giai_doan(df_ohlcv["date"].iloc[i], chuoi_giai_doan, giai_doan_loc):
            continue
        tong_so_phien_quet += 1
        close_i = float(closes.iloc[i])
        do_lech = (close_i - duong_i) / duong_i * 100  # CÓ DẤU: âm = dưới đường, dương = trên đường

        nhom = None
        if khoang_1[0] <= do_lech < khoang_1[1]:
            nhom = 1
        elif khoang_2[0] <= do_lech < khoang_2[1]:
            nhom = 2
        if nhom is None:
            continue

        close_sau = float(closes.iloc[i + so_phien_du_bao])
        pct_thay_doi = (close_sau - close_i) / close_i * 100
        ket_qua_theo_khoang[nhom].append(pct_thay_doi)

    n1, n2 = len(ket_qua_theo_khoang[1]), len(ket_qua_theo_khoang[2])
    if n1 < 5 or n2 < 5:
        return {
            "hop_le": False,
            "so_lan_khoang_1": n1, "so_lan_khoang_2": n2,
            "tong_so_phien_quet": tong_so_phien_quet,
            "ghi_chu": (
                f"Cỡ mẫu quá nhỏ để kiểm định đáng tin cậy (khoảng 1: {n1} lần, "
                f"khoảng 2: {n2} lần) — cần tối thiểu 5 lần mỗi khoảng, khuyến nghị "
                f"tối thiểu 30 lần để kết quả kiểm định có ý nghĩa thực tế."
            ),
        }

    x1 = sum(1 for v in ket_qua_theo_khoang[1] if v > 0)
    x2 = sum(1 for v in ket_qua_theo_khoang[2] if v > 0)
    p1, p2 = x1 / n1, x2 / n2

    p_pool = (x1 + x2) / (n1 + n2)
    se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) ** 0.5

    if se == 0:
        return {
            "hop_le": False,
            "so_lan_khoang_1": n1, "so_lan_khoang_2": n2,
            "tong_so_phien_quet": tong_so_phien_quet,
            "ghi_chu": "Không tính được kiểm định (2 nhóm có tỷ lệ thắng giống hệt nhau, sai số chuẩn bằng 0).",
        }

    z = (p1 - p2) / se
    p_value = float(2 * (1 - norm.cdf(abs(z))))
    co_y_nghia = bool(p_value < nguong_y_nghia)

    return {
        "hop_le": True,
        "so_lan_khoang_1": n1, "so_lan_khoang_2": n2,
        "tong_so_phien_quet": tong_so_phien_quet,
        "xac_suat_thang_khoang_1_pct": round(p1 * 100, 1),
        "xac_suat_thang_khoang_2_pct": round(p2 * 100, 1),
        "pct_thay_doi_trung_binh_khoang_1": round(sum(ket_qua_theo_khoang[1]) / n1, 2),
        "pct_thay_doi_trung_binh_khoang_2": round(sum(ket_qua_theo_khoang[2]) / n2, 2),
        "z_score": round(z, 3),
        "p_value": round(p_value, 4),
        "co_y_nghia_thong_ke": co_y_nghia,
        "nguong_y_nghia": nguong_y_nghia,
        "ghi_chu": (
            f"P-value = {p_value:.4f} {'<' if co_y_nghia else '>='} {nguong_y_nghia} → "
            + (
                "CÓ sự khác biệt ý nghĩa thống kê giữa 2 khoảng (khó xảy ra do ngẫu nhiên)."
                if co_y_nghia else
                "KHÔNG đủ bằng chứng để kết luận có khác biệt ý nghĩa thống kê — chênh "
                "lệch quan sát được CÓ THỂ chỉ do ngẫu nhiên."
            )
        ),
    }


def so_sanh_2_khoang_rsi(
    df_ohlcv: pd.DataFrame,
    khoang_1: tuple[float, float],
    khoang_2: tuple[float, float],
    so_phien_du_bao: int = 5,
    so_phien_kiem_tra: int = 250,
    nguong_y_nghia: float = 0.05,
    chuoi_giai_doan: Optional[pd.Series] = None,
    giai_doan_loc: Optional[str] = None,
) -> dict:
    """Kiểm định thống kê "Kiểm định có RSI (14)" (bổ sung 04/08/2026) —
    TƯƠNG TỰ `so_sanh_2_khoang_do_lech()` ở trên, nhưng phân nhóm theo
    GIÁ TRỊ RSI(14) tại từng thời điểm (thay vì % độ lệch so với đường
    tham chiếu EMA200/MA20). Trả lời câu hỏi kiểu: "RSI dưới 30 (quá
    bán) có khác gì RSI trên 69 (quá mua) về xác suất thắng sau N phiên
    không" — bằng kiểm định 2 tỷ lệ (two-proportion z-test), không chỉ
    so sánh cảm tính 2 con số %.

    `chuoi_giai_doan` + `giai_doan_loc` (bổ sung 05/08/2026): nếu truyền
    cả 2, CHỈ tính các lần trong quá khứ mà NGÀY đó khớp đúng giai đoạn
    thị trường/ngành mong muốn (xem
    `core.market_regime_detector.tinh_chuoi_giai_doan_theo_ngay`).

    `khoang_1`, `khoang_2`: tuple (từ, đến) tính theo GIÁ TRỊ RSI(14)
    (0-100), nửa khoảng đóng-mở [từ, đến). Ví dụ 3 vùng RSI kinh điển:
    Quá bán = (0, 30), Trung tính = (30, 70), Quá mua = (70, 101) — có
    thể chọn BẤT KỲ 2 trong số này (hoặc khoảng tùy ý) để so sánh.

    Cùng phương pháp/ngưỡng ý nghĩa thống kê như `so_sanh_2_khoang_do_lech()`
    — xem docstring hàm đó để biết chi tiết diễn giải p-value.
    """
    from scipy.stats import norm

    if khoang_1[0] >= khoang_1[1]:
        raise InvalidEntryScreenerError('khoang_1: giá trị "từ" phải nhỏ hơn "đến".')
    if khoang_2[0] >= khoang_2[1]:
        raise InvalidEntryScreenerError('khoang_2: giá trị "từ" phải nhỏ hơn "đến".')

    so_phien_toi_thieu = 20  # RSI(14) cần ít dữ liệu hơn nhiều so với EMA200
    if df_ohlcv is None or df_ohlcv.empty or len(df_ohlcv) < so_phien_toi_thieu:
        return {"hop_le": False, "ghi_chu": f"Chưa đủ dữ liệu lịch sử (cần tối thiểu {so_phien_toi_thieu} phiên)."}

    from core.indicators import calculate_rsi

    n = len(df_ohlcv)
    rsi_series = calculate_rsi(df_ohlcv, period=14)
    closes = df_ohlcv["close"]

    start = min(so_phien_toi_thieu, max(0, n - so_phien_kiem_tra))
    end = n - so_phien_du_bao

    ket_qua_theo_khoang = {1: [], 2: []}
    tong_so_phien_quet = 0
    for i in range(start, max(start, end)):
        rsi_i = rsi_series.iloc[i]
        if pd.isna(rsi_i):
            continue
        if not _khop_giai_doan(df_ohlcv["date"].iloc[i], chuoi_giai_doan, giai_doan_loc):
            continue
        tong_so_phien_quet += 1
        rsi_i = float(rsi_i)

        nhom = None
        if khoang_1[0] <= rsi_i < khoang_1[1]:
            nhom = 1
        elif khoang_2[0] <= rsi_i < khoang_2[1]:
            nhom = 2
        if nhom is None:
            continue

        close_i = float(closes.iloc[i])
        close_sau = float(closes.iloc[i + so_phien_du_bao])
        pct_thay_doi = (close_sau - close_i) / close_i * 100
        ket_qua_theo_khoang[nhom].append(pct_thay_doi)

    n1, n2 = len(ket_qua_theo_khoang[1]), len(ket_qua_theo_khoang[2])
    if n1 < 5 or n2 < 5:
        return {
            "hop_le": False,
            "so_lan_khoang_1": n1, "so_lan_khoang_2": n2,
            "tong_so_phien_quet": tong_so_phien_quet,
            "ghi_chu": (
                f"Cỡ mẫu quá nhỏ để kiểm định đáng tin cậy (khoảng 1: {n1} lần, "
                f"khoảng 2: {n2} lần) — cần tối thiểu 5 lần mỗi khoảng, khuyến nghị "
                f"tối thiểu 30 lần để kết quả kiểm định có ý nghĩa thực tế."
            ),
        }

    x1 = sum(1 for v in ket_qua_theo_khoang[1] if v > 0)
    x2 = sum(1 for v in ket_qua_theo_khoang[2] if v > 0)
    p1, p2 = x1 / n1, x2 / n2

    p_pool = (x1 + x2) / (n1 + n2)
    se = (p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) ** 0.5

    if se == 0:
        return {
            "hop_le": False,
            "so_lan_khoang_1": n1, "so_lan_khoang_2": n2,
            "tong_so_phien_quet": tong_so_phien_quet,
            "ghi_chu": "Không tính được kiểm định (2 nhóm có tỷ lệ thắng giống hệt nhau, sai số chuẩn bằng 0).",
        }

    z = (p1 - p2) / se
    p_value = float(2 * (1 - norm.cdf(abs(z))))
    co_y_nghia = bool(p_value < nguong_y_nghia)

    return {
        "hop_le": True,
        "so_lan_khoang_1": n1, "so_lan_khoang_2": n2,
        "tong_so_phien_quet": tong_so_phien_quet,
        "xac_suat_thang_khoang_1_pct": round(p1 * 100, 1),
        "xac_suat_thang_khoang_2_pct": round(p2 * 100, 1),
        "pct_thay_doi_trung_binh_khoang_1": round(sum(ket_qua_theo_khoang[1]) / n1, 2),
        "pct_thay_doi_trung_binh_khoang_2": round(sum(ket_qua_theo_khoang[2]) / n2, 2),
        "z_score": round(z, 3),
        "p_value": round(p_value, 4),
        "co_y_nghia_thong_ke": co_y_nghia,
        "nguong_y_nghia": nguong_y_nghia,
        "ghi_chu": (
            f"P-value = {p_value:.4f} {'<' if co_y_nghia else '>='} {nguong_y_nghia} → "
            + (
                "CÓ sự khác biệt ý nghĩa thống kê giữa 2 khoảng RSI (khó xảy ra do ngẫu nhiên)."
                if co_y_nghia else
                "KHÔNG đủ bằng chứng để kết luận có khác biệt ý nghĩa thống kê giữa 2 khoảng "
                "RSI — chênh lệch quan sát được CÓ THỂ chỉ do ngẫu nhiên."
            )
        ),
    }


# ==============================================================================
# MÔ PHỎNG GIAO DỊCH KHÔNG CHỒNG LẤN THỜI GIAN (bổ sung 06/08/2026) —
# trả lời chính xác "thực tế sẽ có bao nhiêu giao dịch ĐỘC LẬP, lãi/lỗ
# THẬT là bao nhiêu" — KHÁC với tinh_thong_ke_tang_giam_lich_su() (đếm
# TẤT CẢ các lần quan sát, dù CHỒNG LẤN thời gian với nhau).
# ==============================================================================

def mo_phong_giao_dich_khong_chong_lap(
    df_ohlcv: pd.DataFrame,
    tieu_chi_dat: list[str],
    so_phien_giu: int = 4,
    so_phien_kiem_tra: int = 250,
    duong_tham_chieu: str = "ema200",
    chuoi_giai_doan: Optional[pd.Series] = None,
    giai_doan_loc: Optional[str] = None,
    von_ban_dau: float = 1_000_000_000,
    ty_trong_von_moi_lenh: float = 0.5,
) -> dict:
    """Mô phỏng THỰC TẾ số giao dịch ĐỘC LẬP (KHÔNG CHỒNG LẤN thời gian —
    mỗi giao dịch phải "xong" đủ `so_phien_giu` phiên mới được tính giao
    dịch tiếp theo, đúng logic chu kỳ T+ của thị trường VN) mà CHÍNH mã
    này từng thỏa tiêu chí trong quá khứ — sau đó CỘNG DỒN (compound)
    LỢI NHUẬN THẬT (không phải số trung bình giả định) qua ĐÚNG THỨ TỰ
    THỜI GIAN thật đã xảy ra, để biết chính xác NẾU giao dịch từ đầu với
    `von_ban_dau`, dùng `ty_trong_von_moi_lenh` mỗi lần, thì vốn cuối là
    bao nhiêu.

    KHÁC với `tinh_thong_ke_tang_giam_lich_su()`: hàm đó đếm TẤT CẢ các
    lần quan sát dù CHỒNG LẤN nhau (VD ngày 1-5, ngày 2-6... đều tính
    riêng) — phù hợp để tính XÁC SUẤT/Kelly (cần cỡ mẫu lớn). Hàm NÀY
    chỉ đếm các giao dịch THỰC SỰ CÓ THỂ THỰC HIỆN ĐỘC LẬP theo đúng
    ràng buộc thời gian thật — phù hợp để biết SỐ TIỀN THẬT, không phải
    xác suất.

    Đây vẫn là MÔ PHỎNG trên dữ liệu lịch sử, KHÔNG phải cam kết lợi
    nhuận hay dự báo tương lai — chỉ cho biết "nếu quá khứ lặp lại đúng
    như vậy" thì kết quả sẽ ra sao.
    """
    if not (0 < ty_trong_von_moi_lenh <= 1):
        raise InvalidEntryScreenerError("ty_trong_von_moi_lenh phải trong khoảng (0, 1].")
    if von_ban_dau <= 0:
        raise InvalidEntryScreenerError("von_ban_dau phải > 0.")

    tieu_chi_su_dung = [t for t in tieu_chi_dat if t in {"dieu_kien_nen_ema200", "tich_luy_dai_han"}]
    if not tieu_chi_su_dung:
        return {
            "so_giao_dich": 0,
            "ghi_chu": (
                "Mã này chỉ đạt tiêu chí 'Mô hình thu hẹp biên độ' và/hoặc "
                "'Khối lượng breakout' — 2 tiêu chí này KHÔNG được phát lại "
                "trong mô phỏng (quá chậm để tính hàng loạt)."
            ),
        }

    so_phien_toi_thieu = 200 if duong_tham_chieu == "ema200" else 20
    if df_ohlcv is None or df_ohlcv.empty or len(df_ohlcv) < so_phien_toi_thieu:
        return {"so_giao_dich": 0, "ghi_chu": f"Chưa đủ dữ liệu lịch sử (cần tối thiểu {so_phien_toi_thieu} phiên)."}

    from core.indicators import calculate_ema, calculate_ma

    n = len(df_ohlcv)
    duong_series = (
        calculate_ema(df_ohlcv, period=200) if duong_tham_chieu == "ema200"
        else calculate_ma(df_ohlcv, period=20)
    )
    closes = df_ohlcv["close"]

    start = min(so_phien_toi_thieu, max(0, n - so_phien_kiem_tra))
    end = n - so_phien_giu

    giao_dich: list[dict] = []
    i = start
    while i < max(start, end):
        duong_i = duong_series.iloc[i]
        if pd.isna(duong_i) or duong_i <= 0:
            i += 1
            continue
        close_i = float(closes.iloc[i])

        dat_dieu_kien = False
        if "dieu_kien_nen_ema200" in tieu_chi_su_dung:
            do_lech = (close_i - duong_i) / duong_i * 100
            if do_lech >= -10.0:
                dat_dieu_kien = True
        if not dat_dieu_kien and "tich_luy_dai_han" in tieu_chi_su_dung:
            tich_luy = kiem_tra_tich_luy_dai_han(df_ohlcv.iloc[: i + 1])
            if tich_luy.get("dat"):
                dat_dieu_kien = True

        if not dat_dieu_kien or not _khop_giai_doan(df_ohlcv["date"].iloc[i], chuoi_giai_doan, giai_doan_loc):
            i += 1
            continue

        # KHỚP điều kiện -> mở 1 giao dịch ĐỘC LẬP, "khóa" luôn
        # `so_phien_giu` phiên tiếp theo (không cho mở giao dịch mới
        # chồng lấn trong khoảng thời gian này — đúng ràng buộc T+ thật).
        close_sau = float(closes.iloc[i + so_phien_giu])
        pct_thay_doi = (close_sau - close_i) / close_i * 100
        giao_dich.append({
            "ngay_vao": str(df_ohlcv["date"].iloc[i]),
            "ngay_ra": str(df_ohlcv["date"].iloc[i + so_phien_giu]),
            "pct_thay_doi": round(pct_thay_doi, 2),
        })
        i += so_phien_giu  # nhảy hẳn qua khoảng đã "khóa", KHÔNG chồng lấn

    so_giao_dich = len(giao_dich)
    if so_giao_dich == 0:
        return {"so_giao_dich": 0, "ghi_chu": "Không tìm thấy giao dịch độc lập nào trong lịch sử đã quét."}

    # --- Cộng dồn (compound) lợi nhuận THẬT theo ĐÚNG thứ tự thời gian ---
    von = von_ban_dau
    lich_su_von: list[float] = []
    so_lan_thang = 0
    for gd in giao_dich:
        so_tien_dat = von * ty_trong_von_moi_lenh
        lai_lo = so_tien_dat * gd["pct_thay_doi"] / 100
        von += lai_lo
        gd["von_sau_giao_dich"] = round(von)
        lich_su_von.append(von)
        if gd["pct_thay_doi"] > 0:
            so_lan_thang += 1

    return {
        "so_giao_dich": so_giao_dich,
        "so_lan_thang": so_lan_thang,
        "so_lan_thua": so_giao_dich - so_lan_thang,
        "von_ban_dau": von_ban_dau,
        "von_cuoi_cung": round(von),
        "lai_lo_tuyet_doi": round(von - von_ban_dau),
        "lai_lo_pct": round((von - von_ban_dau) / von_ban_dau * 100, 2),
        "chi_tiet_giao_dich": giao_dich,
        "ghi_chu": (
            f"Mô phỏng {so_giao_dich} giao dịch ĐỘC LẬP (không chồng lấn thời gian, "
            f"mỗi giao dịch giữ đúng {so_phien_giu} phiên) — dùng {ty_trong_von_moi_lenh*100:.0f}% "
            f"vốn hiện có cho mỗi lần, cộng dồn lợi nhuận THẬT theo đúng thứ tự lịch sử."
        ),
    }
