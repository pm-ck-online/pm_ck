"""
long_term_indicator_backtest.py
================================
[Bổ sung — Bộ lọc "Cổ phiếu dài hạn"]

So sánh 8 bộ chỉ số kỹ thuật (chiến lược long/flat) cho MỘT mã, tách kết
quả theo giai đoạn thị trường Uptrend/Sideway/Downtrend đang có hiệu lực
tại ngày VÀO LỆNH của từng giao dịch trong lịch sử — dùng để xếp hạng "bộ
chỉ số nào phù hợp nhất với mã này trong giai đoạn hiện tại".

KHÔNG tự viết lại engine backtest — tái sử dụng nguyên vẹn
`backtest.backtest_engine.run_backtest()` (đã test kỹ, thực thi lệnh tại
giá MỞ CỬA ngày kế tiếp để tránh lookahead bias, tự quản lý trạng thái vị
thế và tính phí) và `make_crossover_signals()` cho 2 chiến lược dạng giao
cắt. Đây THUẦN TÚY là công cụ backtest lịch sử — KHÔNG đặt lệnh giao dịch
thật dưới bất kỳ hình thức nào.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from backtest.backtest_engine import Trade, make_crossover_signals, run_backtest
from core.indicators import (
    calculate_bollinger_bands,
    calculate_ema,
    calculate_ma,
    calculate_rsi,
    calculate_volume_ma,
)

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
VALID_REGIMES = {"uptrend", "sideway", "downtrend"}

DEFAULT_INITIAL_CAPITAL = 1_000_000_000.0
DEFAULT_FEE_PCT = 0.15


class InvalidLongTermBacktestError(ValueError):
    """Dữ liệu đầu vào không hợp lệ cho module backtest 'Cổ phiếu dài hạn'."""


def _validate_df(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise InvalidLongTermBacktestError(
            f"DataFrame thiếu các cột bắt buộc: {sorted(missing)}. "
            f"Cần đủ: {sorted(REQUIRED_COLUMNS)}."
        )


# ==============================================================================
# BƯỚC 1 — Tính chỉ báo kỹ thuật cần dùng
# ==============================================================================

def tinh_chi_bao_dai_han(df: pd.DataFrame) -> pd.DataFrame:
    """Trả về bản sao `df` (đã sắp theo ngày tăng dần) kèm các cột chỉ báo
    cần cho 8 bộ chỉ số bên dưới: ma20, ema50, ema200, rsi14, bb_upper/
    bb_middle/bb_lower, vol_ma20.
    """
    _validate_df(df)
    df = df.sort_values("date").reset_index(drop=True).copy()

    df["ma20"] = calculate_ma(df, 20)
    df["ema50"] = calculate_ema(df, 50)
    df["ema200"] = calculate_ema(df, 200)
    df["rsi14"] = calculate_rsi(df, 14)
    upper, middle, lower = calculate_bollinger_bands(df, 20, 2.0)
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = upper, middle, lower
    df["vol_ma20"] = calculate_volume_ma(df, 20)
    return df


# ==============================================================================
# BƯỚC 2 — 8 bộ chỉ số (mỗi bộ = 1 cặp Series tín hiệu vào/ra, KHÔNG cần
# tự quản lý trạng thái "đang giữ hàng" — run_backtest() đã làm việc đó).
# ==============================================================================

def xay_8_bo_chi_so(df_co_chi_bao: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Trả về `{tên_bộ_chỉ_số: (entry_series, exit_series)}` — mỗi Series
    kiểu bool, cùng độ dài/thứ tự với `df_co_chi_bao` (đã gọi
    `tinh_chi_bao_dai_han()` trước đó).
    """
    df = df_co_chi_bao
    close, volume = df["close"], df["volume"]

    ma20_entry, ma20_exit = make_crossover_signals(close, df["ma20"])
    ema_entry, ema_exit = make_crossover_signals(df["ema50"], df["ema200"])

    rsi_entry = (df["rsi14"] < 30).fillna(False)
    rsi_exit = (df["rsi14"] > 70).fillna(False)

    bb_breakout_entry = ((close > df["bb_upper"]) & (volume > 1.2 * df["vol_ma20"])).fillna(False)
    bb_breakout_exit = (close < df["bb_middle"]).fillna(False)

    bb_bounce_entry = (close <= df["bb_lower"]).fillna(False)
    bb_bounce_exit = (close >= df["bb_middle"]).fillna(False)

    vol_breakout_entry = ((close > df["ma20"]) & (volume > 1.5 * df["vol_ma20"])).fillna(False)
    vol_breakout_exit = (close < df["ma20"]).fillna(False)

    trend_up = df["ema50"] > df["ema200"]
    trend_filter_entry = ((df["rsi14"] < 35) & trend_up).fillna(False)
    trend_filter_exit = ((df["rsi14"] > 70) | (~trend_up)).fillna(False)

    valid_ema200 = df["ema200"].notna()
    buy_hold_entry = (valid_ema200 & ~valid_ema200.shift(1, fill_value=False)).fillna(False)
    buy_hold_exit = pd.Series(False, index=df.index)

    return {
        "MA20 (Giá cắt MA20)": (ma20_entry, ma20_exit),
        "EMA50/EMA200 (Golden/Death Cross)": (ema_entry, ema_exit),
        "RSI14 (Quá mua/Quá bán 30-70)": (rsi_entry, rsi_exit),
        "Bollinger Breakout + Volume": (bb_breakout_entry, bb_breakout_exit),
        "Bollinger Bounce (mua đáy dải dưới)": (bb_bounce_entry, bb_bounce_exit),
        "Volume Breakout + MA20": (vol_breakout_entry, vol_breakout_exit),
        "Kết hợp: Trend Filter EMA + RSI": (trend_filter_entry, trend_filter_exit),
        "Mua và giữ (Buy & Hold)": (buy_hold_entry, buy_hold_exit),
    }


# ==============================================================================
# BƯỚC 3 — Backtest 1 bộ chỉ số, bucket kết quả theo giai đoạn tại ngày
# vào lệnh của từng giao dịch.
# ==============================================================================

def _mo_phong_von_tuan_tu(trades: list[Trade], initial_capital: float) -> dict:
    """Mô phỏng lại vốn TUẦN TỰ chỉ với các lệnh trong `trades` (đã lọc
    theo 1 giai đoạn cụ thể) — coi như đây là TOÀN BỘ lịch sử giao dịch,
    bắt đầu lại từ `initial_capital`. Dùng `trade.pnl_pct` (đã tính đúng
    phí 2 chiều bởi `run_backtest()`) làm tỷ suất sinh lời từng lệnh.
    """
    if not trades:
        return {
            "n_trades": 0, "win_rate_pct": None, "total_return_pct": None,
            "avg_return_pct": None, "ending_capital": initial_capital,
        }
    capital = initial_capital
    wins = 0
    for t in trades:
        capital *= (1 + t.pnl_pct / 100.0)
        if t.pnl > 0:
            wins += 1
    return {
        "n_trades": len(trades),
        "win_rate_pct": wins / len(trades) * 100.0,
        "total_return_pct": (capital / initial_capital - 1.0) * 100.0,
        "avg_return_pct": float(np.mean([t.pnl_pct for t in trades])),
        "ending_capital": capital,
    }


def backtest_theo_giai_doan(
    df_co_chi_bao: pd.DataFrame,
    regime_series: pd.Series,
    entry_series: pd.Series,
    exit_series: pd.Series,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    fee_pct: float = DEFAULT_FEE_PCT,
) -> dict[str, dict]:
    """Chạy `run_backtest()` 1 lần cho 1 bộ chỉ số, rồi bucket các lệnh
    (`BacktestResult.trades`) theo giai đoạn (`regime_series`, index=ngày,
    giá trị "uptrend"/"sideway"/"downtrend") có hiệu lực tại NGÀY VÀO LỆNH
    của từng lệnh.

    Trả về `{giai_đoạn: {n_trades, win_rate_pct, total_return_pct,
    avg_return_pct, ending_capital}}` cho cả 3 giai đoạn (giá trị mặc định
    khi giai đoạn không có lệnh nào: `n_trades=0`, các % là `None`).
    """
    result = run_backtest(
        df_co_chi_bao,
        entry_signal_fn=lambda _df: entry_series,
        exit_signal_fn=lambda _df: exit_series,
        initial_cash=initial_capital,
        fee_pct=fee_pct,
    )

    regime_tai_ngay = regime_series.to_dict()
    trades_theo_giai_doan: dict[str, list[Trade]] = {r: [] for r in VALID_REGIMES}
    for t in result.trades:
        regime = regime_tai_ngay.get(pd.Timestamp(t.entry_date))
        if regime in trades_theo_giai_doan:
            trades_theo_giai_doan[regime].append(t)

    return {
        regime: _mo_phong_von_tuan_tu(trades, initial_capital)
        for regime, trades in trades_theo_giai_doan.items()
    }


def backtest_toan_bo_8_bo_chi_so(
    df: pd.DataFrame,
    regime_series: pd.Series,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    fee_pct: float = DEFAULT_FEE_PCT,
) -> dict[str, dict[str, dict]]:
    """Tính chỉ báo + backtest CẢ 8 bộ chỉ số cho 1 mã, trả về
    `{tên_bộ_chỉ_số: {giai_đoạn: {...}}}`.
    """
    df_co_chi_bao = tinh_chi_bao_dai_han(df)
    bo_chi_so = xay_8_bo_chi_so(df_co_chi_bao)
    return {
        ten: backtest_theo_giai_doan(
            df_co_chi_bao, regime_series, entry, exit_,
            initial_capital=initial_capital, fee_pct=fee_pct,
        )
        for ten, (entry, exit_) in bo_chi_so.items()
    }


def tim_bo_chi_so_tot_nhat(ket_qua: dict[str, dict[str, dict]], regime: str) -> Optional[str]:
    """Trong kết quả trả về từ `backtest_toan_bo_8_bo_chi_so()`, tìm tên bộ
    chỉ số có `total_return_pct` cao nhất cho 1 `regime`, CHỈ xét các bộ có
    ít nhất 1 lệnh (`n_trades > 0`). Trả về `None` nếu không có bộ nào có
    lệnh trong giai đoạn này.
    """
    if regime not in VALID_REGIMES:
        raise InvalidLongTermBacktestError(
            f"Giai đoạn '{regime}' không hợp lệ. Phải là 1 trong {sorted(VALID_REGIMES)}."
        )
    ung_vien = [
        (ten, ket_qua_giai_doan[regime]["total_return_pct"])
        for ten, ket_qua_giai_doan in ket_qua.items()
        if ket_qua_giai_doan[regime]["n_trades"] > 0
    ]
    if not ung_vien:
        return None
    return max(ung_vien, key=lambda x: x[1])[0]
