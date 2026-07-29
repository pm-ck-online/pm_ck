"""
stock_signal_engine.py
=========================
[Bổ sung — Module Chỉ tiêu Khuyến nghị Mua/Bán Cổ phiếu]

Module cuối cùng trong chuỗi: Điểm Vĩ Mô -> Trạng thái Thị trường -> Phân
bổ Vốn -> **Tín hiệu MUA/GIỮ/BÁN từng mã cụ thể**.

Với MỖI MÃ, hệ thống xuất ra 1 trong 3 trạng thái: MUA | GIU_THEO_DOI |
BAN, kèm điểm số tổng hợp và danh sách lý do cụ thể.

GIỚI HẠN DỮ LIỆU QUAN TRỌNG: dự án CHƯA có nguồn dữ liệu tài chính doanh
nghiệp (EPS, ROE, D/E, CFO, P/E ngành...) — các hàm `evaluate_fundamental_*`
nhận `fundamentals` là THAM SỐ TÙY CHỌN; nếu không truyền, phần lọc cơ
bản sẽ được BỎ QUA (coi như trung tính) và module chỉ hoạt động dựa trên
tín hiệu KỸ THUẬT — khác với tài liệu gốc yêu cầu đủ cả 2 lớp. Cần bổ
sung nguồn dữ liệu tài chính (Vietstock/CafeF/vnstock Company API) trước
khi dùng lớp cơ bản cho quyết định thực tế.

KHÔNG phải khuyến nghị đầu tư cá nhân hóa hay tín hiệu giao dịch tự động
— người dùng chịu trách nhiệm với quyết định giao dịch của mình.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


class InvalidStockSignalError(ValueError):
    """Dữ liệu đầu vào không hợp lệ cho module tín hiệu mua/bán."""


# ==============================================================================
# NHẬN DIỆN MẪU NẾN ĐẢO CHIỀU (đơn giản hóa, không dùng TA-Lib)
# ==============================================================================

def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    """Nến Bullish Engulfing tại PHIÊN CUỐI CÙNG: nến trước giảm (đỏ),
    nến sau tăng (xanh) và thân nến sau "nuốt trọn" thân nến trước.
    """
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    prev_bearish = prev["close"] < prev["open"]
    curr_bullish = curr["close"] > curr["open"]
    engulfs = curr["open"] <= prev["close"] and curr["close"] >= prev["open"]
    return bool(prev_bearish and curr_bullish and engulfs)


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    """Đối xứng `is_bullish_engulfing` — báo hiệu đảo chiều giảm."""
    if len(df) < 2:
        return False
    prev, curr = df.iloc[-2], df.iloc[-1]
    prev_bullish = prev["close"] > prev["open"]
    curr_bearish = curr["close"] < curr["open"]
    engulfs = curr["open"] >= prev["close"] and curr["close"] <= prev["open"]
    return bool(prev_bullish and curr_bearish and engulfs)


def is_pin_bar(df: pd.DataFrame, direction: str = "bullish") -> bool:
    """Nến Pin Bar (thân nhỏ, bóng dài 1 phía) tại phiên cuối cùng.

    `direction="bullish"`: bóng dưới dài (>= 2x thân), gợi ý từ chối giá thấp.
    `direction="bearish"`: bóng trên dài (>= 2x thân), gợi ý từ chối giá cao.
    """
    if len(df) < 1:
        return False
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    if body == 0:
        body = 1e-9  # tránh chia cho 0 với nến doji tuyệt đối
    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]

    if direction == "bullish":
        return bool(lower_wick >= 2 * body and upper_wick < body)
    if direction == "bearish":
        return bool(upper_wick >= 2 * body and lower_wick < body)
    raise InvalidStockSignalError("direction phải là 'bullish' hoặc 'bearish'.")


# ==============================================================================
# ĐIỀU KIỆN KỸ THUẬT KÍCH HOẠT MUA (mục 2.2 tài liệu)
# ==============================================================================

def check_base_trend_condition(close: float, ema200: Optional[float], adx: Optional[float]) -> bool:
    """Điều kiện xu hướng nền bắt buộc: giá trên EMA200 VÀ ADX(14) > 25."""
    if ema200 is None or adx is None:
        return False
    return close > ema200 and adx > 25


def check_pullback_pattern(
    df: pd.DataFrame, ema20: float, ema50: float, rsi_series: pd.Series
) -> bool:
    """Mua theo Pullback: giá hồi về vùng EMA20/EMA50, có nến đảo chiều
    tăng (Engulfing/Pin Bar), VÀ RSI hồi từ vùng 40-50 đi lên.
    """
    last_close = df["close"].iloc[-1]
    near_ema_zone = min(ema20, ema50) * 0.98 <= last_close <= max(ema20, ema50) * 1.02
    has_reversal_candle = is_bullish_engulfing(df) or is_pin_bar(df, direction="bullish")

    if len(rsi_series) < 2:
        rsi_recovering = False
    else:
        prev_rsi, curr_rsi = rsi_series.iloc[-2], rsi_series.iloc[-1]
        rsi_recovering = bool(
            not pd.isna(prev_rsi) and not pd.isna(curr_rsi)
            and 35 <= prev_rsi <= 50 and curr_rsi > prev_rsi
        )

    return bool(near_ema_zone and has_reversal_candle and rsi_recovering)


def check_breakout_pattern(
    df: pd.DataFrame, resistance_level: float, volume_ma20: float, multiplier: float = 1.5
) -> bool:
    """Mua theo Breakout: giá đóng cửa vượt kháng cự VÀ khối lượng phiên
    breakout > `multiplier` lần khối lượng trung bình 20 phiên.
    """
    last = df.iloc[-1]
    price_breaks_resistance = last["close"] > resistance_level
    volume_confirms = volume_ma20 > 0 and last["volume"] > multiplier * volume_ma20
    return bool(price_breaks_resistance and volume_confirms)


def check_support_bounce_pattern(
    df: pd.DataFrame, support_level: float, rsi_series: pd.Series, tolerance_pct: float = 2.0
) -> bool:
    """Mua tại vùng hỗ trợ (Sideway): giá chạm/nảy từ vùng hỗ trợ VÀ
    RSI(14) < 40 đang quay đầu tăng.

    ĐƠN GIẢN HÓA: tài liệu gốc yêu cầu hỗ trợ đã "xác nhận ≥ 2 lần trước
    đó" — việc đếm số lần chạm hỗ trợ lịch sử cần logic phát hiện swing
    phức tạp hơn, CHƯA cài đặt ở bản này; hàm chỉ kiểm tra giá đang ở gần
    mức hỗ trợ ĐÃ CHO TRƯỚC (tham số) và RSI đang hồi phục.
    """
    last_close = df["close"].iloc[-1]
    near_support = abs(last_close - support_level) / support_level * 100 <= tolerance_pct

    if len(rsi_series) < 2:
        rsi_turning_up = False
    else:
        prev_rsi, curr_rsi = rsi_series.iloc[-2], rsi_series.iloc[-1]
        rsi_turning_up = bool(
            not pd.isna(prev_rsi) and not pd.isna(curr_rsi)
            and curr_rsi < 40 and curr_rsi > prev_rsi
        )

    return bool(near_support and rsi_turning_up)


def check_bullish_divergence_pattern(df: pd.DataFrame) -> bool:
    """Mua theo phân kỳ tăng: phát hiện Bullish Divergence VÀ khối lượng
    tại đáy sau THẤP HƠN đáy trước (dấu hiệu bán tháo cạn kiệt).
    """
    from core.market_breadth import detect_bullish_divergence

    result = detect_bullish_divergence(df)
    if not result.get("detected"):
        return False

    date_low_1 = result["date_low_1"]
    date_low_2 = result["date_low_2"]
    vol_at_low1 = df.loc[df["date"] == date_low_1, "volume"]
    vol_at_low2 = df.loc[df["date"] == date_low_2, "volume"]
    if vol_at_low1.empty or vol_at_low2.empty:
        return True  # không đủ dữ liệu khối lượng đối chiếu -> vẫn chấp nhận phân kỳ đơn thuần

    return bool(vol_at_low2.iloc[0] < vol_at_low1.iloc[0])


def evaluate_technical_buy_trigger(
    df: pd.DataFrame,
    ema20: float, ema50: float, ema200: float, adx: float,
    rsi_series: pd.Series,
    resistance_level: Optional[float] = None,
    support_level: Optional[float] = None,
    volume_ma20: Optional[float] = None,
) -> dict:
    """Tổng hợp điều kiện kỹ thuật kích hoạt MUA (mục 2.2). Trả về:
        {"kich_hoat": bool, "mau_hinh": str|None, "ly_do": str}
    """
    last_close = df["close"].iloc[-1]

    if not check_base_trend_condition(last_close, ema200, adx):
        return {
            "kich_hoat": False, "mau_hinh": None,
            "ly_do": "Chưa đủ điều kiện nền (giá dưới EMA200 hoặc ADX < 25).",
        }

    if check_pullback_pattern(df, ema20, ema50, rsi_series):
        return {"kich_hoat": True, "mau_hinh": "PULLBACK", "ly_do": "Giá hồi về EMA20/50 kèm nến đảo chiều và RSI phục hồi."}

    if resistance_level is not None and volume_ma20 is not None:
        if check_breakout_pattern(df, resistance_level, volume_ma20):
            return {"kich_hoat": True, "mau_hinh": "BREAKOUT", "ly_do": "Giá vượt kháng cự kèm khối lượng đột biến."}

    if support_level is not None:
        if check_support_bounce_pattern(df, support_level, rsi_series):
            return {"kich_hoat": True, "mau_hinh": "VUNG_HO_TRO", "ly_do": "Giá nảy từ vùng hỗ trợ, RSI hồi phục từ vùng quá bán."}

    if check_bullish_divergence_pattern(df):
        return {"kich_hoat": True, "mau_hinh": "PHAN_KY_TANG", "ly_do": "Phát hiện phân kỳ tăng, khối lượng bán tháo cạn kiệt."}

    return {"kich_hoat": False, "mau_hinh": None, "ly_do": "Không khớp mẫu hình kích hoạt nào."}


# ==============================================================================
# CHỈ TIÊU BÁN CẮT LỖ (mục 3.1 — ưu tiên cao nhất)
# ==============================================================================

def check_stop_loss_hit(
    current_price: float,
    stop_loss_price: Optional[float] = None,
    recent_low: Optional[float] = None,
    current_loss_pct_nav: Optional[float] = None,
    max_loss_pct_nav: float = 0.02,
) -> bool:
    """Kiểm tra điều kiện BÁN CẮT LỖ — bất kỳ điều kiện nào khớp là kích
    hoạt (mục 3.1). Bỏ qua điều kiện nào không có đủ dữ liệu.
    """
    if stop_loss_price is not None and current_price <= stop_loss_price:
        return True
    if recent_low is not None and current_price < recent_low:
        return True
    if current_loss_pct_nav is not None and current_loss_pct_nav >= max_loss_pct_nav:
        return True
    return False


# ==============================================================================
# CHỈ TIÊU BÁN CHỐT LỜI / KỸ THUẬT (mục 3.2.a)
# ==============================================================================

def check_resistance_overbought(
    df: pd.DataFrame, resistance_level: float, rsi_series: pd.Series, tolerance_pct: float = 2.0
) -> bool:
    """Giá tiệm cận/chạm kháng cự mạnh VÀ RSI(14) > 70 (quá mua)."""
    last_close = df["close"].iloc[-1]
    near_resistance = abs(last_close - resistance_level) / resistance_level * 100 <= tolerance_pct
    if rsi_series.empty or pd.isna(rsi_series.iloc[-1]):
        return False
    return bool(near_resistance and rsi_series.iloc[-1] > 70)


def check_volume_depletion(df: pd.DataFrame, lookback: int = 5) -> bool:
    """Giá vẫn tăng nhưng khối lượng giảm dần liên tục — xu hướng thiếu
    "nhiên liệu" (mục 3.2.a).
    """
    if len(df) < lookback + 1:
        return False
    recent = df.tail(lookback + 1)
    price_rising = recent["close"].iloc[-1] > recent["close"].iloc[0]
    volumes = recent["volume"].tolist()
    volume_declining = all(volumes[i] > volumes[i + 1] for i in range(len(volumes) - 1))
    return bool(price_rising and volume_declining)


def check_ema200_break_confirmed(
    df: pd.DataFrame, ema200_series: pd.Series, adx: Optional[float], sessions: int = 2, adx_threshold: float = 25,
) -> bool:
    """Giá đóng cửa dưới EMA200 liên tiếp `sessions` phiên VÀ ADX(14) > 25
    theo hướng giảm (mục 3.2.a).
    """
    if len(df) < sessions or len(ema200_series) < sessions:
        return False
    recent_closes = df["close"].tail(sessions)
    recent_ema200 = ema200_series.tail(sessions)
    all_below = bool((recent_closes.values < recent_ema200.values).all())
    adx_confirms = adx is not None and adx > adx_threshold
    return bool(all_below and adx_confirms)


def evaluate_technical_sell_trigger(
    df: pd.DataFrame,
    rsi_series: pd.Series,
    ma50_series: pd.Series,
    ma200_series: pd.Series,
    adx: Optional[float] = None,
    resistance_level: Optional[float] = None,
) -> dict:
    """Tổng hợp chỉ tiêu kỹ thuật BÁN CHỐT LỜI (mục 3.2.a). Trả về:
        {"tin_hieu": bool, "ly_do": list[str]}
    """
    from core.market_breadth import detect_bearish_divergence, detect_ma_cross

    ly_do = []

    if resistance_level is not None and check_resistance_overbought(df, resistance_level, rsi_series):
        ly_do.append("Giá chạm kháng cự mạnh, RSI > 70 (quá mua).")

    divergence = detect_bearish_divergence(df)
    if divergence.get("detected"):
        ly_do.append("Phát hiện phân kỳ giảm (Bearish Divergence).")

    if detect_ma_cross(ma50_series, ma200_series) == "death_cross":
        ly_do.append("MA50 cắt xuống dưới MA200 (Death Cross).")

    if check_volume_depletion(df):
        ly_do.append("Giá tăng nhưng khối lượng cạn kiệt liên tục.")

    if check_ema200_break_confirmed(df, ma200_series, adx):
        ly_do.append("Giá gãy xuống dưới EMA200, ADX xác nhận xu hướng giảm.")

    return {"tin_hieu": len(ly_do) > 0, "ly_do": ly_do}


# ==============================================================================
# LỚP CƠ BẢN (tùy chọn — CHƯA có nguồn dữ liệu tài chính thật, xem docstring đầu file)
# ==============================================================================

FUNDAMENTAL_BUY_THRESHOLDS = {
    "eps_growth_yoy_min": 0.10,
    "eps_growth_consecutive_quarters_min": 2,
    "pe_vs_industry_max": 1.0,
    "peg_max": 1.0,
    "roe_min": 0.15,
}

FUNDAMENTAL_SELL_THRESHOLDS = {
    "eps_decline_consecutive_quarters_min": 2,
    "roe_decline_from_peak_pct": 0.30,
    "de_spike_pct": 0.50,
    "pe_vs_industry_overextended": 1.5,
    "peg_overextended": 2.0,
    "cfo_negative_consecutive_quarters_min": 2,
}


def evaluate_fundamental_buy_screen(fundamentals: Optional[dict]) -> dict:
    """Lọc cơ bản cho MUA (mục 2.1). Nếu `fundamentals=None` (chưa có dữ
    liệu tài chính), trả về `dat=None` (TRUNG TÍNH — không chặn cũng
    không xác nhận), kèm ghi chú rõ ràng.

    `fundamentals` kỳ vọng có các khóa: eps_growth_yoy, eps_growth_quarters_streak,
    pe, pe_industry_avg, peg, roe, de, de_industry_avg, cfo, cfo_growth.
    """
    if fundamentals is None:
        return {"dat": None, "ly_do": ["Chưa có dữ liệu tài chính — bỏ qua lớp cơ bản."]}

    checks = {
        "eps_tang_truong": (
            fundamentals.get("eps_growth_yoy", 0) > FUNDAMENTAL_BUY_THRESHOLDS["eps_growth_yoy_min"]
            and fundamentals.get("eps_growth_quarters_streak", 0)
            >= FUNDAMENTAL_BUY_THRESHOLDS["eps_growth_consecutive_quarters_min"]
        ),
        "pe_hop_ly": (
            fundamentals.get("pe_industry_avg", 0) > 0
            and fundamentals.get("pe", float("inf")) / fundamentals["pe_industry_avg"]
            <= FUNDAMENTAL_BUY_THRESHOLDS["pe_vs_industry_max"]
        ),
        "peg_hop_ly": fundamentals.get("peg", float("inf")) < FUNDAMENTAL_BUY_THRESHOLDS["peg_max"],
        "roe_cao": fundamentals.get("roe", 0) > FUNDAMENTAL_BUY_THRESHOLDS["roe_min"],
        "de_kiem_soat": (
            fundamentals.get("de", float("inf")) <= fundamentals.get("de_industry_avg", float("inf"))
        ),
        "cfo_tot": fundamentals.get("cfo", 0) > 0 and fundamentals.get("cfo_growth", 0) >= 0,
    }

    all_pass = all(checks.values())
    ly_do = [k for k, v in checks.items() if v]
    return {"dat": all_pass, "ly_do": ly_do, "chi_tiet": checks}


def evaluate_fundamental_sell_screen(fundamentals: Optional[dict]) -> dict:
    """Lọc cơ bản cho BÁN (mục 3.2.b) — thay đổi luận điểm đầu tư. Nếu
    `fundamentals=None`, trả về `dat=None` (TRUNG TÍNH).
    """
    if fundamentals is None:
        return {"dat": None, "ly_do": ["Chưa có dữ liệu tài chính — bỏ qua lớp cơ bản."]}

    ly_do = []

    if fundamentals.get("eps_decline_quarters_streak", 0) >= FUNDAMENTAL_SELL_THRESHOLDS["eps_decline_consecutive_quarters_min"]:
        ly_do.append("EPS suy giảm liên tiếp ≥ 2 quý.")

    roe_peak = fundamentals.get("roe_peak")
    roe_current = fundamentals.get("roe")
    if roe_peak and roe_current is not None and roe_peak > 0:
        if (roe_peak - roe_current) / roe_peak > FUNDAMENTAL_SELL_THRESHOLDS["roe_decline_from_peak_pct"]:
            ly_do.append("ROE suy giảm mạnh so với đỉnh gần nhất.")

    de_prev = fundamentals.get("de_prev_quarter")
    de_current = fundamentals.get("de")
    if de_prev and de_current is not None and de_prev > 0:
        if (de_current - de_prev) / de_prev > FUNDAMENTAL_SELL_THRESHOLDS["de_spike_pct"]:
            ly_do.append("D/E tăng đột biến so với quý trước.")

    pe = fundamentals.get("pe")
    pe_industry = fundamentals.get("pe_industry_avg")
    peg = fundamentals.get("peg")
    if pe and pe_industry and peg is not None:
        if pe > FUNDAMENTAL_SELL_THRESHOLDS["pe_vs_industry_overextended"] * pe_industry and peg > FUNDAMENTAL_SELL_THRESHOLDS["peg_overextended"]:
            ly_do.append("P/E vượt xa trung bình ngành, tăng trưởng không tương xứng (PEG cao).")

    if fundamentals.get("cfo_negative_quarters_streak", 0) >= FUNDAMENTAL_SELL_THRESHOLDS["cfo_negative_consecutive_quarters_min"]:
        ly_do.append("Dòng tiền kinh doanh (CFO) âm kéo dài dù lợi nhuận vẫn dương.")

    return {"dat": len(ly_do) > 0, "ly_do": ly_do}


# ==============================================================================
# ĐIỀU KIỆN PHỦ QUYẾT MUA (mục 2.3)
# ==============================================================================

def check_buy_veto(
    macro_score: Optional[float],
    market_regime: Optional[str],
    avg_volume_20: Optional[float] = None,
    min_liquidity_threshold: Optional[float] = None,
) -> dict:
    """Kiểm tra điều kiện PHỦ QUYẾT mua (mục 2.3). Trả về
    {"phu_quyet": bool, "ly_do": list[str]}.
    """
    ly_do = []

    if macro_score is not None and market_regime is not None:
        if macro_score < -1.0 and market_regime == "DOWNTREND":
            ly_do.append("Macro Score tiêu cực mạnh (<-1.0) VÀ thị trường đang DOWNTREND.")

    if avg_volume_20 is not None and min_liquidity_threshold is not None:
        if avg_volume_20 < min_liquidity_threshold:
            ly_do.append("Thanh khoản trung bình 20 phiên quá thấp so với quy mô vốn dự kiến.")

    return {"phu_quyet": len(ly_do) > 0, "ly_do": ly_do}


# ==============================================================================
# HÀM CHÍNH — theo đúng BẢNG QUYẾT ĐỊNH 5 BƯỚC (mục 4 tài liệu)
# ==============================================================================

def evaluate_stock_signal(
    symbol: str,
    df: pd.DataFrame,
    macro_score: Optional[float] = None,
    market_regime: Optional[str] = None,
    resistance_level: Optional[float] = None,
    support_level: Optional[float] = None,
    fundamentals: Optional[dict] = None,
    position_info: Optional[dict] = None,
    min_liquidity_threshold: Optional[float] = None,
    strategy: str = "dau_tu",
) -> dict:
    """Hàm CHÍNH — chạy đúng 5 bước quyết định (mục 4 tài liệu) cho MỘT
    mã cổ phiếu, trả về khuyến nghị MUA | GIU_THEO_DOI | BAN.

    `position_info` (nếu đang nắm giữ mã này): {"gia_cat_lo", "day_gan_nhat",
    "loi_lo_hien_tai_pct_nav"} — dùng cho BƯỚC 1 (kiểm tra cắt lỗ).
    `strategy`: "dau_tu" (w_fundamental=0.6) | "giao_dich" (w_fundamental=0.3).

    Output theo đúng cấu trúc mục 5 tài liệu (rút gọn tên khóa sang
    snake_case, giữ nguyên ý nghĩa).
    """
    from core.indicators import calculate_ema, calculate_ma, calculate_rsi

    if df is None or df.empty:
        raise InvalidStockSignalError("df không được rỗng.")

    last_close = float(df["close"].iloc[-1])
    ema20 = calculate_ema(df, 20).iloc[-1]
    ema50 = calculate_ema(df, 50).iloc[-1]
    ema200 = calculate_ema(df, 200).iloc[-1] if len(df) >= 200 else None
    ma50_series = calculate_ma(df, 50)
    ma200_series = calculate_ma(df, 200) if len(df) >= 200 else pd.Series(dtype=float)
    rsi_series = calculate_rsi(df, period=14) if len(df) > 14 else pd.Series(dtype=float)
    volume_ma20 = df["volume"].tail(20).mean() if len(df) >= 20 else None

    adx = None
    try:
        from core.market_breadth import calculate_adx
        adx_series = calculate_adx(df, period=14)
        if not adx_series.empty and not pd.isna(adx_series.iloc[-1]):
            adx = float(adx_series.iloc[-1])
    except Exception:  # noqa: BLE001
        pass

    canh_bao: list[str] = []

    # --- BƯỚC 1: Kiểm tra cắt lỗ (ưu tiên cao nhất, dừng xử lý ngay nếu khớp) ---
    if position_info:
        stop_hit = check_stop_loss_hit(
            current_price=last_close,
            stop_loss_price=position_info.get("gia_cat_lo"),
            recent_low=position_info.get("day_gan_nhat"),
            current_loss_pct_nav=position_info.get("loi_lo_hien_tai_pct_nav"),
        )
        if stop_hit:
            return {
                "ma": symbol, "khuyen_nghi": "BAN", "loai_ban": "CAT_LO", "uu_tien": "CAO",
                "stock_score": None, "fundamental_score": None, "technical_score": None,
                "chi_tiet": {"ly_do": ["Đã chạm điều kiện cắt lỗ — BÁN NGAY, không xét thêm điều kiện khác."]},
                "khoang_gia_vao_lenh_de_xuat": None, "canh_bao": [],
                "ghi_chu": "Bán cắt lỗ được ưu tiên tuyệt đối, không thương lượng.",
            }

    # --- BƯỚC 2: Điều kiện phủ quyết mua ---
    veto = check_buy_veto(macro_score, market_regime, volume_ma20, min_liquidity_threshold)

    # --- BƯỚC 3: Bán chốt lời ---
    sell_technical = evaluate_technical_sell_trigger(
        df, rsi_series, ma50_series, ma200_series, adx, resistance_level,
    )
    sell_fundamental = evaluate_fundamental_sell_screen(fundamentals)

    if sell_fundamental["dat"]:
        return {
            "ma": symbol, "khuyen_nghi": "BAN", "loai_ban": "CHOT_LOI", "uu_tien": "CAO",
            "stock_score": None, "fundamental_score": None, "technical_score": None,
            "chi_tiet": {"co_ban": sell_fundamental["ly_do"], "ky_thuat": sell_technical["ly_do"]},
            "khoang_gia_vao_lenh_de_xuat": None, "canh_bao": [],
            "ghi_chu": "Bán do THAY ĐỔI LUẬN ĐIỂM đầu tư (cơ bản) — ưu tiên cao.",
        }
    if sell_technical["tin_hieu"]:
        return {
            "ma": symbol, "khuyen_nghi": "BAN", "loai_ban": "CHOT_LOI", "uu_tien": "TRUNG_BINH",
            "stock_score": None, "fundamental_score": None, "technical_score": None,
            "chi_tiet": {"ky_thuat": sell_technical["ly_do"]},
            "khoang_gia_vao_lenh_de_xuat": None, "canh_bao": [],
            "ghi_chu": "Tín hiệu bán kỹ thuật — có thể chờ xác nhận thêm 1 phiên.",
        }

    # --- BƯỚC 4: Kiểm tra điều kiện mua (nếu không bị phủ quyết) ---
    buy_fundamental = evaluate_fundamental_buy_screen(fundamentals)
    buy_technical = evaluate_technical_buy_trigger(
        df, ema20, ema50, ema200, adx, rsi_series, resistance_level, support_level, volume_ma20,
    )

    if veto["phu_quyet"]:
        canh_bao.extend(veto["ly_do"])
        khuyen_nghi = "GIU_THEO_DOI"
    else:
        fundamental_ok = buy_fundamental["dat"] is not False  # None (chưa có data) hoặc True đều KHÔNG chặn
        if fundamental_ok and buy_technical["kich_hoat"]:
            khuyen_nghi = "MUA"
        elif fundamental_ok and buy_fundamental["dat"] is True:
            khuyen_nghi = "GIU_THEO_DOI"  # đủ cơ bản, chưa đủ kỹ thuật -> chờ
        else:
            khuyen_nghi = "GIU_THEO_DOI"

    # --- Tính điểm tổng hợp (đơn giản hóa: quy đổi tín hiệu kỹ thuật/cơ bản có/không thành thang -1..+1) ---
    w_fundamental, w_technical = (0.6, 0.4) if strategy == "dau_tu" else (0.3, 0.7)
    technical_score = 1.0 if buy_technical["kich_hoat"] else (-1.0 if sell_technical["tin_hieu"] else 0.0)
    if buy_fundamental["dat"] is None:
        fundamental_score = None
    else:
        fundamental_score = 1.0 if buy_fundamental["dat"] else -1.0
    stock_score = (
        w_technical * technical_score
        if fundamental_score is None
        else w_fundamental * fundamental_score + w_technical * technical_score
    )

    entry_range = None
    if khuyen_nghi == "MUA" and resistance_level is not None:
        entry_range = [round(resistance_level * 1.001, 2), round(resistance_level * 1.01, 2)]

    return {
        "ma": symbol,
        "khuyen_nghi": khuyen_nghi,
        "loai_ban": None,
        "uu_tien": None,
        "stock_score": round(stock_score, 3),
        "fundamental_score": fundamental_score,
        "technical_score": technical_score,
        "chi_tiet": {
            "co_ban_dat": buy_fundamental["ly_do"],
            "ky_thuat_dat": [buy_technical["ly_do"]] if buy_technical["kich_hoat"] else [],
            "mau_hinh_ky_thuat": buy_technical["mau_hinh"],
        },
        "khoang_gia_vao_lenh_de_xuat": entry_range,
        "canh_bao": canh_bao,
        "ghi_chu": "Cần đối chiếu Module Phân bổ Vốn để xác định khối lượng và mức cắt lỗ cụ thể.",
    }


# ==============================================================================
# BÁO CÁO TỔNG HỢP — danh sách mã đủ điều kiện MUA/BÁN (bước cuối theo yêu cầu)
# ==============================================================================

def build_signal_summary_report(evaluations: list[dict]) -> dict:
    """Tổng hợp danh sách kết quả `evaluate_stock_signal()` của NHIỀU MÃ
    thành báo cáo: nhóm theo khuyến nghị (MUA/BÁN/GIỮ), sắp xếp mã MUA
    theo `stock_score` giảm dần, mã BÁN CẮT LỖ lên đầu (khẩn cấp nhất).

    Trả về:
        {"mua": [...], "ban_cat_lo": [...], "ban_chot_loi": [...],
         "giu_theo_doi": [...], "tong_so_ma": int}
    """
    mua = [e for e in evaluations if e["khuyen_nghi"] == "MUA"]
    ban_cat_lo = [e for e in evaluations if e["khuyen_nghi"] == "BAN" and e.get("loai_ban") == "CAT_LO"]
    ban_chot_loi = [e for e in evaluations if e["khuyen_nghi"] == "BAN" and e.get("loai_ban") == "CHOT_LOI"]
    giu = [e for e in evaluations if e["khuyen_nghi"] == "GIU_THEO_DOI"]

    mua_sorted = sorted(mua, key=lambda e: e.get("stock_score") or 0, reverse=True)
    ban_chot_loi_sorted = sorted(
        ban_chot_loi, key=lambda e: 0 if e.get("uu_tien") == "CAO" else 1,
    )

    return {
        "mua": mua_sorted,
        "ban_cat_lo": ban_cat_lo,
        "ban_chot_loi": ban_chot_loi_sorted,
        "giu_theo_doi": giu,
        "tong_so_ma": len(evaluations),
    }
