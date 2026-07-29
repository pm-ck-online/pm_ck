"""
indicators.py
=============
[Giai đoạn 2 — Chỉ báo kỹ thuật]

Tính các chỉ báo kỹ thuật trên DataFrame OHLCV
(cột bắt buộc: date, open, high, low, close, volume).

CHỈ GỒM CÁC CHỈ BÁO CHÍNH đã chốt trong yêu cầu dự án — không thêm chỉ báo
khác để tránh phức tạp hóa không cần thiết:

1. MA20  — trung bình động ĐƠN GIẢN (SMA) chu kỳ 20 phiên, tín hiệu ngắn hạn.
2. EMA50, EMA100 — xu hướng trung hạn.
3. EMA200 — xu hướng dài hạn; đây là đường QUAN TRỌNG NHẤT trong toàn hệ
   thống vì `market_regime_detector` dùng vị trí giá so với EMA200 làm căn
   cứ chính để xác định giai đoạn thị trường.
4. Volume trung bình 15-20 phiên — dùng để xác nhận breakout (volume đột
   biến > 1.5-2 lần trung bình) và nhận diện vùng tích lũy (volume thấp,
   ổn định quanh mức trung bình).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


class InsufficientDataError(ValueError):
    """Không đủ dữ liệu (số phiên) để tính chỉ báo với chu kỳ yêu cầu."""


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
# CÁC HÀM TÍNH CHỈ BÁO
# ==============================================================================

def calculate_ma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Tính trung bình động ĐƠN GIẢN (Simple Moving Average).

    Dùng cho MA20 — tín hiệu ngắn hạn theo yêu cầu dự án.
    """
    _validate_df(df)
    if column not in df.columns:
        raise ValueError(f"Cột '{column}' không tồn tại trong DataFrame.")
    return df[column].rolling(window=period, min_periods=period).mean()


def calculate_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Tính trung bình động lũy thừa (Exponential Moving Average).

    Dùng cho EMA50, EMA100 (trung hạn) và EMA200 (dài hạn — đường chuẩn
    xác nhận giai đoạn thị trường trong market_regime_detector).
    """
    _validate_df(df)
    if column not in df.columns:
        raise ValueError(f"Cột '{column}' không tồn tại trong DataFrame.")
    # adjust=False: công thức EMA "chuẩn" theo cách tính phổ biến trong
    # phân tích kỹ thuật (giống các nền tảng như TradingView).
    return df[column].ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_volume_ma(df: pd.DataFrame, period: int) -> pd.Series:
    """Tính trung bình động đơn giản của khối lượng giao dịch (volume).

    Dùng cho volume_ma_15 / volume_ma_20 — xác nhận breakout và nhận diện
    vùng tích lũy (volume thấp, ổn định quanh mức trung bình).
    """
    _validate_df(df)
    return df["volume"].rolling(window=period, min_periods=period).mean()


def is_volume_breakout(
    df: pd.DataFrame, multiplier: float = 1.5, volume_ma_period: int = 20
) -> bool:
    """Kiểm tra phiên gần nhất có phải breakout theo volume hay không.

    Trả về True nếu volume phiên hiện tại > `multiplier` lần volume trung
    bình `volume_ma_period` phiên trước đó (không bao gồm phiên hiện tại).
    """
    _validate_df(df, min_rows=volume_ma_period + 1)

    current_volume = df["volume"].iloc[-1]
    # Lấy volume MA của các phiên TRƯỚC phiên hiện tại, để không tự so
    # sánh volume hiện tại với chính nó.
    prior_volume_ma = df["volume"].iloc[-(volume_ma_period + 1):-1].mean()

    if prior_volume_ma == 0 or pd.isna(prior_volume_ma):
        return False

    return bool(current_volume > multiplier * prior_volume_ma)


def get_indicator_snapshot(
    df: pd.DataFrame, config: Optional[dict] = None
) -> dict:
    """Tính toàn bộ chỉ báo tại phiên gần nhất, trả về dạng dict để các
    module khác (market_regime_detector, pattern_detector) dùng trực tiếp.

    Tham số `config` có thể truyền các chu kỳ tùy chỉnh (khớp với
    `config.yaml` mục `indicators`); nếu không truyền, dùng giá trị mặc
    định đã thống nhất trong dự án (MA20, EMA50/100/200, volume MA 15/20).

    Trả về dict gồm:
        - date: ngày của phiên gần nhất
        - close: giá đóng cửa phiên gần nhất
        - ma20, ema50, ema100, ema200
        - volume, volume_ma_15, volume_ma_20
        - price_above_ema200: bool — vị trí giá so với EMA200 (dùng trực
          tiếp bởi market_regime_detector ở Bước 2 của thuật toán)
        - is_volume_breakout: bool — volume phiên gần nhất có đột biến
          so với trung bình 20 phiên hay không.

    Các giá trị chỉ báo có thể là None nếu chưa đủ dữ liệu lịch sử để tính
    (ví dụ chưa đủ 200 phiên cho EMA200) — các module gọi hàm này cần tự
    kiểm tra None trước khi sử dụng.
    """
    _validate_df(df, min_rows=1)
    cfg = config or {}

    ma_short_period = cfg.get("ma_short_period", 20)
    ema_mid_periods = cfg.get("ema_mid_periods", [50, 100])
    ema_long_period = cfg.get("ema_long_period", 200)
    volume_ma_periods = cfg.get("volume_ma_periods", [15, 20])
    breakout_multiplier = cfg.get("breakout_volume_multiplier", 1.5)

    def _last_or_none(series: pd.Series):
        if len(series) == 0:
            return None
        value = series.iloc[-1]
        return None if pd.isna(value) else float(value)

    ma20 = _last_or_none(calculate_ma(df, ma_short_period))
    ema_mid = {
        f"ema{p}": _last_or_none(calculate_ema(df, p)) for p in ema_mid_periods
    }
    ema200 = _last_or_none(calculate_ema(df, ema_long_period))

    volume_ma = {
        f"volume_ma_{p}": _last_or_none(calculate_volume_ma(df, p))
        for p in volume_ma_periods
    }

    last_close = float(df["close"].iloc[-1])
    last_volume = float(df["volume"].iloc[-1])
    last_date = df["date"].iloc[-1]

    price_above_ema200 = None if ema200 is None else bool(last_close > ema200)

    breakout_period = max(volume_ma_periods) if volume_ma_periods else 20
    try:
        volume_breakout = is_volume_breakout(
            df, multiplier=breakout_multiplier, volume_ma_period=breakout_period
        )
    except InsufficientDataError:
        volume_breakout = None

    snapshot = {
        "date": last_date,
        "close": last_close,
        "volume": last_volume,
        "ma20": ma20,
        "ema200": ema200,
        "price_above_ema200": price_above_ema200,
        "is_volume_breakout": volume_breakout,
    }
    snapshot.update(ema_mid)
    snapshot.update(volume_ma)
    return snapshot


# ==============================================================================
# RSI (Relative Strength Index) — chỉ báo BỔ SUNG theo yêu cầu riêng
# ==============================================================================
# LƯU Ý: Đây KHÔNG nằm trong bộ chỉ báo cốt lõi ban đầu của dự án (vốn chỉ
# gồm MA20/EMA50/EMA100/EMA200/volume MA theo đúng yêu cầu gốc). Hàm này
# được thêm bổ sung theo yêu cầu cụ thể, dùng độc lập cho mục đích hiển
# thị biểu đồ (dashboard) — KHÔNG được đưa vào `get_indicator_snapshot()`
# ở trên để tránh làm thay đổi cấu trúc output mà các module khác
# (market_regime_detector, pattern_detector...) đã phụ thuộc và được test.

def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.Series:
    """Tính chỉ số sức mạnh tương đối RSI (Relative Strength Index).

    Dùng công thức làm mượt kiểu Wilder (Wilder's smoothing) — cách tính
    RSI phổ biến nhất, khớp với mặc định của TradingView/fireant.

    Công thức:
        RS  = trung bình tăng (làm mượt) / trung bình giảm (làm mượt)
        RSI = 100 - 100 / (1 + RS)

    Trả về pd.Series giá trị trong khoảng [0, 100]. `period` phiên đầu
    tiên sẽ là NaN (chưa đủ dữ liệu làm mượt).
    """
    _validate_df(df, min_rows=period + 1)

    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Làm mượt kiểu Wilder ~ tương đương EMA với alpha = 1/period.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Trường hợp avg_loss = 0 (giá chỉ tăng liên tục) -> RSI = 100.
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


# ==============================================================================
# GỘP KHUNG THỜI GIAN (resample) — dùng cho biểu đồ chọn Ngày/Tuần/Tháng
# ==============================================================================

TIMEFRAME_RULES = {
    "day": "D",
    "week": "W-FRI",   # tuần kết thúc vào Thứ 6 — khớp lịch giao dịch VN
    "month": "ME",     # tháng (month-end)
}


def resample_ohlcv(df: pd.DataFrame, timeframe: str = "day") -> pd.DataFrame:
    """Gộp dữ liệu OHLCV theo ngày thành khung thời gian lớn hơn (tuần,
    tháng) — dùng cho việc HIỂN THỊ biểu đồ ở nhiều khung thời gian mà
    KHÔNG cần gọi thêm request nào tới nguồn dữ liệu (chỉ gộp lại dữ liệu
    ngày đã có sẵn).

    `timeframe`: "day" (giữ nguyên) | "week" (gộp theo tuần, kết thúc Thứ
    6) | "month" (gộp theo tháng).

    Quy tắc gộp:
        - open  = giá mở cửa phiên ĐẦU TIÊN trong kỳ
        - high  = giá cao nhất trong kỳ
        - low   = giá thấp nhất trong kỳ
        - close = giá đóng cửa phiên CUỐI CÙNG trong kỳ
        - volume = TỔNG khối lượng trong kỳ
    """
    _validate_df(df)
    if timeframe not in TIMEFRAME_RULES:
        raise ValueError(
            f"timeframe '{timeframe}' không hợp lệ. Cần một trong "
            f"{sorted(TIMEFRAME_RULES.keys())}."
        )

    if timeframe == "day":
        return df.sort_values("date").reset_index(drop=True)

    rule = TIMEFRAME_RULES[timeframe]
    df_indexed = df.sort_values("date").set_index("date")

    resampled = df_indexed.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    resampled = resampled.dropna(subset=["open"]).reset_index()
    return resampled
