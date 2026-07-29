"""
market_breadth.py
====================
[Bổ sung — Mô hình 3 lớp xác định trạng thái thị trường]

Module này cài đặt LỚP 2 (độ rộng thị trường theo EMA200 — lớp quyết
định chính) và LỚP 3 (các chỉ báo xác nhận bổ sung: MA50/MA200 cross,
ADX, độ rộng Bollinger Band, Volume Ratio, đường A/D, tỷ lệ đỉnh/đáy 52
tuần) theo đúng công thức trong tài liệu kỹ thuật đã cung cấp.

NGUYÊN TẮC: Lớp 3 CHỈ dùng để xác nhận/cảnh báo mâu thuẫn — KHÔNG tự
quyết định nhãn trạng thái thị trường (nhãn do Lớp 2 quyết định, sau khi
đã bị Lớp 1 - vĩ mô - giới hạn trần).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


class InsufficientDataError(ValueError):
    """Không đủ dữ liệu để tính chỉ báo với chu kỳ yêu cầu."""


def _validate_df(df: pd.DataFrame, min_rows: int = 1) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame thiếu các cột bắt buộc: {sorted(missing)}. "
            f"Cần đủ: {sorted(REQUIRED_COLUMNS)}."
        )
    if len(df) < min_rows:
        raise InsufficientDataError(
            f"Cần tối thiểu {min_rows} phiên dữ liệu, nhưng chỉ có {len(df)}."
        )


# ==============================================================================
# LỚP 2 — ĐỘ RỘNG THỊ TRƯỜNG THEO EMA200 (market breadth)
# ==============================================================================

def calculate_ema200_breadth(snapshots: list[dict]) -> dict:
    """Tính % mã trong nhóm đang đóng cửa TRÊN EMA200 của chính mã đó.

    `snapshots`: danh sách dict, mỗi phần tử tối thiểu có 'close', 'ema200'
    (ví dụ lấy từ `core.indicators.get_indicator_snapshot()`). Mã chưa đủ
    dữ liệu tính EMA200 (ema200=None) sẽ bị LOẠI khỏi mẫu số, đúng theo
    yêu cầu "tránh làm méo kết quả".

    Trả về: {"breadth_pct": float|None, "n_above": int, "n_valid": int}.
    """
    valid = [s for s in snapshots if s.get("ema200") is not None and s.get("close") is not None]
    if not valid:
        return {"breadth_pct": None, "n_above": 0, "n_valid": 0}

    n_above = sum(1 for s in valid if s["close"] > s["ema200"])
    breadth_pct = n_above / len(valid) * 100.0
    return {"breadth_pct": breadth_pct, "n_above": n_above, "n_valid": len(valid)}


def calculate_ema200_deviation(snapshots: list[dict]) -> Optional[float]:
    """Độ lệch trung bình (%) của giá so với EMA200 trên toàn nhóm.

    Độ lệch(mã) = (Giá - EMA200) / EMA200 * 100%
    Trả về trung bình cộng của độ lệch này trên các mã có đủ dữ liệu, hoặc
    None nếu không có mã nào hợp lệ.
    """
    valid = [s for s in snapshots if s.get("ema200") not in (None, 0) and s.get("close") is not None]
    if not valid:
        return None

    deviations = [(s["close"] - s["ema200"]) / s["ema200"] * 100.0 for s in valid]
    return sum(deviations) / len(deviations)


BREADTH_THRESHOLDS = {
    "uptrend_extreme": 80.0,
    "uptrend": 60.0,
    "sideway_low": 40.0,
    "downtrend": 40.0,
    "downtrend_extreme": 20.0,
}


def classify_breadth_label(breadth_pct: Optional[float], breadth_trend: Optional[str] = None) -> str:
    """Phân loại nhãn trạng thái theo bảng ngưỡng (mục 2.4 tài liệu):

        > 80%              -> "uptrend_extreme" (quá mua diện rộng)
        > 60% và đang tăng  -> "uptrend"
        40% - 60%           -> "sideway"
        < 40% và đang giảm  -> "downtrend"
        < 20%               -> "downtrend_extreme" (vùng washout)

    `breadth_trend`: "increasing" | "decreasing" | "flat" | None — dùng để
    phân biệt "> 60%" (uptrend) khỏi trường hợp biên chưa rõ hướng. Nếu
    không truyền, mặc định coi là đủ điều kiện theo đúng ngưỡng % thuần
    túy (bỏ qua yếu tố xu hướng).

    LƯU Ý: các ngưỡng 60/40/20/80 là điểm khởi đầu tham khảo theo thông
    lệ quốc tế — CẦN backtest lại trên dữ liệu lịch sử VN-Index để hiệu
    chỉnh trước khi dùng cho quyết định thực tế (xem ghi chú trong tài
    liệu kỹ thuật gốc).
    """
    if breadth_pct is None:
        return "sideway"  # không đủ dữ liệu -> mặc định trung tính, an toàn

    if breadth_pct > BREADTH_THRESHOLDS["uptrend_extreme"]:
        return "uptrend_extreme"
    if breadth_pct > BREADTH_THRESHOLDS["uptrend"]:
        if breadth_trend == "decreasing":
            return "sideway"  # trên 60% nhưng đang giảm dần -> chưa đủ tin cậy là uptrend
        return "uptrend"
    if breadth_pct < BREADTH_THRESHOLDS["downtrend_extreme"]:
        return "downtrend_extreme"
    if breadth_pct < BREADTH_THRESHOLDS["downtrend"]:
        if breadth_trend == "increasing":
            return "sideway"
        return "downtrend"
    return "sideway"


def calculate_breadth_trend(breadth_history: list[float], lookback: int = 10) -> Optional[str]:
    """Xác định xu hướng của chuỗi % Breadth qua `lookback` phiên gần
    nhất: "increasing" | "decreasing" | "flat".

    So sánh giá trị đầu và cuối của cửa sổ `lookback` phiên — đơn giản,
    dễ diễn giải, đúng tinh thần "theo dõi chuỗi thời gian tối thiểu
    10-20 phiên" trong tài liệu gốc.
    """
    if len(breadth_history) < 2:
        return None

    window = breadth_history[-lookback:]
    change = window[-1] - window[0]

    if change > 2.0:
        return "increasing"
    if change < -2.0:
        return "decreasing"
    return "flat"


# ==============================================================================
# LỚP 3 — CÁC CHỈ BÁO XÁC NHẬN BỔ SUNG (không tự quyết định nhãn)
# ==============================================================================

def detect_ma_cross(fast: pd.Series, slow: pd.Series) -> str:
    """Phát hiện Golden Cross / Death Cross tại điểm GẦN NHẤT của 2 chuỗi
    MA (thường MA50 vs MA200).

    Trả về: "golden_cross" | "death_cross" | "none".
    """
    if len(fast) < 2 or len(slow) < 2:
        return "none"

    fast = fast.reset_index(drop=True)
    slow = slow.reset_index(drop=True)

    prev_fast, prev_slow = fast.iloc[-2], slow.iloc[-2]
    curr_fast, curr_slow = fast.iloc[-1], slow.iloc[-1]

    if pd.isna(prev_fast) or pd.isna(prev_slow) or pd.isna(curr_fast) or pd.isna(curr_slow):
        return "none"

    if prev_fast <= prev_slow and curr_fast > curr_slow:
        return "golden_cross"
    if prev_fast >= prev_slow and curr_fast < curr_slow:
        return "death_cross"
    return "none"


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Tính ATR (Average True Range) theo làm mượt kiểu Wilder — dùng làm
    thước đo biến động để tính khoảng giá vào lệnh/cắt lỗ/chốt lời trong
    `core/capital_allocation_engine.py`.

        True Range = Max[High-Low, |High-Close(t-1)|, |Low-Close(t-1)|]
        ATR = EMA_Wilder(True Range, period)
    """
    _validate_df(df, min_rows=period + 1)

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Tính ADX (Average Directional Index) theo đúng công thức Wilder
    (mục 2.5.b tài liệu gốc):

        +DM, -DM từ chênh lệch High/Low giữa 2 phiên liên tiếp
        TR = True Range
        +DI, -DI = 100 * EMA_Wilder(+DM hoặc -DM) / EMA_Wilder(TR)
        DX  = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = EMA_Wilder(DX)

    EMA kiểu Wilder tương đương ewm(alpha=1/period, adjust=False).
    """
    _validate_df(df, min_rows=period + 1)

    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return adx


def calculate_bollinger_band_width(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> pd.Series:
    """Tính độ rộng dải Bollinger (%) theo đúng công thức (mục 2.5.c):

        SMA20, Dải trên/dưới = SMA20 +/- num_std * độ lệch chuẩn(period)
        Band Width (%) = (Dải trên - Dải dưới) / SMA20 * 100%
    """
    _validate_df(df, min_rows=period)

    sma = df["close"].rolling(window=period, min_periods=period).mean()
    std = df["close"].rolling(window=period, min_periods=period).std()

    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width_pct = (upper - lower) / sma * 100.0
    return band_width_pct


def calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Volume Ratio = khối lượng phiên hiện tại / trung bình `period`
    phiên gần nhất (mục 2.5.d).
    """
    _validate_df(df, min_rows=period)
    volume_ma = df["volume"].rolling(window=period, min_periods=period).mean()
    return df["volume"] / volume_ma


def calculate_advance_decline_line(daily_price_changes: list[pd.Series]) -> pd.Series:
    """Tính đường Advance-Decline (A/D Line) tích lũy toàn thị trường
    (mục 2.5.e).

    `daily_price_changes`: danh sách các pd.Series (mỗi phần tử là chuỗi
    % thay đổi giá hàng ngày của MỘT mã, cùng độ dài/cùng chỉ số ngày).

    A/D(t) = A/D(t-1) + (Số mã tăng giá phiên t - Số mã giảm giá phiên t)
    """
    if not daily_price_changes:
        raise ValueError("daily_price_changes không được rỗng.")

    changes_df = pd.concat(daily_price_changes, axis=1)
    advances = (changes_df > 0).sum(axis=1)
    declines = (changes_df < 0).sum(axis=1)
    net = advances - declines
    return net.cumsum()


def aggregate_layer3_indicators_for_group(
    ohlcv_by_symbol: dict[str, "pd.DataFrame"],
) -> dict:
    """Tổng hợp chỉ báo Lớp 3 (MA cross, ADX, Band Width percentile) cho
    MỘT NHÓM/NGÀNH gồm NHIỀU MÃ, dựa trên OHLCV của từng mã đã có sẵn
    (không cần gọi thêm API).

    Quy tắc tổng hợp:
        - MA cross: biểu quyết ĐA SỐ giữa các mã (golden_cross/death_cross/
          none) — chỉ chọn golden/death nếu THỰC SỰ chiếm đa số rõ ràng so
          với cả 2 lựa chọn còn lại, nếu không rõ ràng thì "none".
        - ADX, Band Width percentile: TRUNG BÌNH CỘNG giữa các mã có đủ
          dữ liệu tính toán (bỏ qua mã thiếu dữ liệu, không làm hỏng kết
          quả chung).

    Trả về dict tương thích trực tiếp với tham số `layer3_indicators` của
    `core.market_regime_detector.detect_market_regime_quant()`.
    """
    from core.indicators import calculate_ma

    ma_cross_votes = {"golden_cross": 0, "death_cross": 0, "none": 0}
    adx_values: list[float] = []
    band_width_percentiles: list[float] = []

    for symbol, df in ohlcv_by_symbol.items():
        try:
            ma50 = calculate_ma(df, 50)
            ma200 = calculate_ma(df, 200)
            cross = detect_ma_cross(ma50, ma200)
            ma_cross_votes[cross] = ma_cross_votes.get(cross, 0) + 1
        except Exception:  # noqa: BLE001
            pass

        try:
            adx_series = calculate_adx(df, period=14)
            if not adx_series.empty and not pd.isna(adx_series.iloc[-1]):
                adx_values.append(float(adx_series.iloc[-1]))
        except Exception:  # noqa: BLE001
            pass

        try:
            bw_series = calculate_bollinger_band_width(df, period=20)
            bw_valid = bw_series.dropna()
            if len(bw_valid) >= 20:
                percentile = (bw_valid <= bw_valid.iloc[-1]).mean() * 100.0
                band_width_percentiles.append(float(percentile))
        except Exception:  # noqa: BLE001
            pass

    result: dict = {}

    top_label = max(ma_cross_votes, key=lambda k: ma_cross_votes[k])
    top_count = ma_cross_votes[top_label]
    other_counts = [v for k, v in ma_cross_votes.items() if k != top_label]
    if top_label != "none" and other_counts and top_count > max(other_counts):
        result["ma_cross"] = top_label
    else:
        result["ma_cross"] = "none"

    if adx_values:
        result["adx"] = sum(adx_values) / len(adx_values)
    if band_width_percentiles:
        result["band_width_percentile"] = sum(band_width_percentiles) / len(band_width_percentiles)

    return result


def calculate_new_high_low_ratio(
    close_prices_by_symbol: dict[str, pd.Series], window: int = 252
) -> dict:
    """Tính tỷ lệ mã tạo đỉnh/đáy 52 tuần (mặc định `window=252` phiên ~
    1 năm giao dịch) mới TẠI PHIÊN GẦN NHẤT (mục 2.5.f).

    `close_prices_by_symbol`: dict {symbol: pd.Series giá đóng cửa}, mỗi
    Series đã sắp xếp theo thời gian tăng dần.

    Trả về: {"new_high_ratio": float, "new_low_ratio": float,
             "n_new_high": int, "n_new_low": int, "n_symbols": int}.
    """
    if not close_prices_by_symbol:
        raise ValueError("close_prices_by_symbol không được rỗng.")

    n_new_high = 0
    n_new_low = 0
    n_valid = 0

    for symbol, prices in close_prices_by_symbol.items():
        if len(prices) < 2:
            continue
        n_valid += 1
        window_prices = prices.tail(window)
        latest = prices.iloc[-1]
        if latest >= window_prices.max():
            n_new_high += 1
        if latest <= window_prices.min():
            n_new_low += 1

    if n_valid == 0:
        return {
            "new_high_ratio": 0.0, "new_low_ratio": 0.0,
            "n_new_high": 0, "n_new_low": 0, "n_symbols": 0,
        }

    return {
        "new_high_ratio": n_new_high / n_valid * 100.0,
        "new_low_ratio": n_new_low / n_valid * 100.0,
        "n_new_high": n_new_high,
        "n_new_low": n_new_low,
        "n_symbols": n_valid,
    }


# ==============================================================================
# BỔ SUNG — PHÁT HIỆN PHÂN KỲ TĂNG (Bullish Divergence) trên RSI
# ==============================================================================
# Dùng để "mở khóa" khuyến nghị mua trong giai đoạn DOWNTREND (xem
# `core/capital_allocation_engine.py` — nguyên tắc "không bắt đáy khi
# chưa có tín hiệu phân kỳ tăng rõ ràng").
#
# Phân kỳ tăng cổ điển: giá tạo ĐÁY SAU THẤP HƠN đáy trước, nhưng RSI tại
# đáy sau lại CAO HƠN đáy trước — cho thấy đà giảm đang yếu dần, một dấu
# hiệu cảnh báo sớm khả năng đảo chiều tăng.

def _find_local_minima_indices(values, order: int = 3) -> list[int]:
    """Tìm chỉ số các điểm "đáy swing" — thấp hơn hoặc bằng tất cả điểm
    trong cửa sổ `order` phiên trước và sau nó. Bỏ qua các điểm liền kề
    trùng lặp (do đáy bị "phẳng" nhiều phiên) — chỉ giữ điểm ĐẦU TIÊN của
    mỗi cụm đáy.
    """
    n = len(values)
    minima: list[int] = []
    for i in range(order, n - order):
        window = values[i - order: i + order + 1]
        if values[i] == min(window):
            if not minima or i - minima[-1] > order:
                minima.append(i)
    return minima


def detect_bullish_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    lookback: int = 90,
    swing_order: int = 3,
) -> dict:
    """Phát hiện phân kỳ tăng (Bullish Divergence) giữa giá và RSI, dựa
    trên 2 đáy swing GẦN NHẤT trong `lookback` phiên gần nhất.

    LƯU Ý QUAN TRỌNG VỀ ĐỘ TRỄ: một đáy swing chỉ được XÁC NHẬN sau khi
    đã có đủ `swing_order` phiên SAU nó (để biết chắc đó là đáy, không bị
    phá tiếp) — nghĩa là tín hiệu này có độ trễ tự nhiên khoảng
    `swing_order` phiên so với thời điểm đáy thực sự hình thành. Đây là
    đánh đổi cần thiết giữa tốc độ phát hiện và độ tin cậy.

    Trả về dict:
        {"detected": bool, "reason": str (nếu không đủ dữ liệu),
         "price_low_1", "price_low_2", "rsi_low_1", "rsi_low_2",
         "date_low_1", "date_low_2"} (các khóa giá trị chỉ có khi
        `detected` có thể xác định được, tức đủ dữ liệu).
    """
    from core.indicators import calculate_rsi as _calculate_rsi

    min_rows_needed = rsi_period + swing_order * 2 + 1
    if len(df) < min_rows_needed:
        return {
            "detected": False,
            "reason": f"Cần tối thiểu {min_rows_needed} phiên dữ liệu, chỉ có {len(df)}.",
        }

    rsi = _calculate_rsi(df, period=rsi_period)

    recent_df = df.tail(lookback).reset_index(drop=True)
    recent_rsi = rsi.tail(lookback).reset_index(drop=True)

    lows = recent_df["low"].tolist()
    swing_indices = _find_local_minima_indices(lows, order=swing_order)

    if len(swing_indices) < 2:
        return {
            "detected": False,
            "reason": "Không đủ 2 đáy swing đã xác nhận để so sánh phân kỳ.",
        }

    idx1, idx2 = swing_indices[-2], swing_indices[-1]
    price_low1, price_low2 = lows[idx1], lows[idx2]
    rsi_low1, rsi_low2 = recent_rsi.iloc[idx1], recent_rsi.iloc[idx2]

    if pd.isna(rsi_low1) or pd.isna(rsi_low2):
        return {
            "detected": False,
            "reason": "RSI chưa đủ dữ liệu tại (các) điểm đáy swing được phát hiện.",
        }

    is_lower_low = price_low2 < price_low1
    is_higher_rsi_low = rsi_low2 > rsi_low1
    detected = bool(is_lower_low and is_higher_rsi_low)

    return {
        "detected": detected,
        "price_low_1": float(price_low1),
        "price_low_2": float(price_low2),
        "rsi_low_1": float(rsi_low1),
        "rsi_low_2": float(rsi_low2),
        "date_low_1": recent_df["date"].iloc[idx1],
        "date_low_2": recent_df["date"].iloc[idx2],
    }


def _find_local_maxima_indices(values, order: int = 3) -> list[int]:
    """Tìm chỉ số các điểm "đỉnh swing" — đối xứng với `_find_local_minima_indices`."""
    n = len(values)
    maxima: list[int] = []
    for i in range(order, n - order):
        window = values[i - order: i + order + 1]
        if values[i] == max(window):
            if not maxima or i - maxima[-1] > order:
                maxima.append(i)
    return maxima


def detect_bearish_divergence(
    df: pd.DataFrame,
    rsi_period: int = 14,
    lookback: int = 90,
    swing_order: int = 3,
) -> dict:
    """Phát hiện phân kỳ giảm (Bearish Divergence) — ĐỐI XỨNG với
    `detect_bullish_divergence()`: giá tạo ĐỈNH SAU CAO HƠN đỉnh trước,
    nhưng RSI tại đỉnh sau lại THẤP HƠN đỉnh trước — cho thấy đà tăng
    đang yếu dần, cảnh báo sớm khả năng đảo chiều giảm. Dùng cho tín hiệu
    BÁN chốt lời (mục 3.2.a tài liệu Stock Signal Engine).

    Cùng lưu ý về độ trễ xác nhận đỉnh swing như `detect_bullish_divergence`.
    """
    from core.indicators import calculate_rsi as _calculate_rsi

    min_rows_needed = rsi_period + swing_order * 2 + 1
    if len(df) < min_rows_needed:
        return {
            "detected": False,
            "reason": f"Cần tối thiểu {min_rows_needed} phiên dữ liệu, chỉ có {len(df)}.",
        }

    rsi = _calculate_rsi(df, period=rsi_period)

    recent_df = df.tail(lookback).reset_index(drop=True)
    recent_rsi = rsi.tail(lookback).reset_index(drop=True)

    highs = recent_df["high"].tolist()
    swing_indices = _find_local_maxima_indices(highs, order=swing_order)

    if len(swing_indices) < 2:
        return {
            "detected": False,
            "reason": "Không đủ 2 đỉnh swing đã xác nhận để so sánh phân kỳ.",
        }

    idx1, idx2 = swing_indices[-2], swing_indices[-1]
    price_high1, price_high2 = highs[idx1], highs[idx2]
    rsi_high1, rsi_high2 = recent_rsi.iloc[idx1], recent_rsi.iloc[idx2]

    if pd.isna(rsi_high1) or pd.isna(rsi_high2):
        return {
            "detected": False,
            "reason": "RSI chưa đủ dữ liệu tại (các) điểm đỉnh swing được phát hiện.",
        }

    is_higher_high = price_high2 > price_high1
    is_lower_rsi_high = rsi_high2 < rsi_high1
    detected = bool(is_higher_high and is_lower_rsi_high)

    return {
        "detected": detected,
        "price_high_1": float(price_high1),
        "price_high_2": float(price_high2),
        "rsi_high_1": float(rsi_high1),
        "rsi_high_2": float(rsi_high2),
        "date_high_1": recent_df["date"].iloc[idx1],
        "date_high_2": recent_df["date"].iloc[idx2],
    }
