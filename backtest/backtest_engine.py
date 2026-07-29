"""
backtest_engine.py
====================
[Giai đoạn 2 — Backtest]

Kiểm chứng chỉ báo/mô hình trên dữ liệu lịch sử thị trường chứng khoán
Việt Nam. Đây THUẦN TÚY là công cụ kiểm chứng — KHÔNG kết nối lệnh thật
dưới bất kỳ hình thức nào.

CÁCH HOẠT ĐỘNG:
- Nhận vào 1 DataFrame OHLCV + 2 hàm tín hiệu (entry_signal_fn,
  exit_signal_fn), mỗi hàm nhận df và trả về pd.Series kiểu bool cùng độ
  dài với df, đánh dấu ngày nào có tín hiệu vào/ra lệnh.
- Tín hiệu được ĐÁNH GIÁ tại phiên đóng cửa ngày i, nhưng LỆNH ĐƯỢC THỰC
  THI tại giá MỞ CỬA ngày i+1 — tránh lỗi "nhìn trước tương lai"
  (lookahead bias) khi backtest.
- Chỉ quản lý MỘT vị thế tại một thời điểm (phù hợp việc backtest 1 mã).
- Có tính phí giao dịch (mua + bán) theo % cấu hình được.
- Nếu còn vị thế mở khi hết dữ liệu, TỰ ĐỘNG đóng vị thế ở giá đóng cửa
  phiên cuối cùng để báo cáo kết quả cuối cùng là đầy đủ (không để lại
  lãi/lỗ chưa ghi nhận).

Xuất báo cáo: tổng lợi nhuận, tỷ lệ thắng, Sharpe ratio, max drawdown, số
lệnh, và biểu đồ equity curve (matplotlib).

Hỗ trợ walk-forward: chia dữ liệu thành nhiều giai đoạn train/test theo
thời gian để tránh đánh giá quá lạc quan (overfitting) trên một giai đoạn
duy nhất.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "open", "close"}

TRADING_DAYS_PER_YEAR = 252


# ==============================================================================
# CẤU TRÚC DỮ LIỆU KẾT QUẢ
# ==============================================================================

@dataclass
class Trade:
    """Một lệnh mua-bán hoàn chỉnh (round-trip) trong quá trình backtest."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    entry_fee: float
    exit_fee: float
    pnl: float          # lãi/lỗ tuyệt đối (VND), đã trừ phí 2 chiều
    pnl_pct: float       # lãi/lỗ theo % trên vốn đã bỏ ra cho lệnh này
    forced_close: bool = False  # True nếu bị đóng cưỡng bức do hết dữ liệu


@dataclass
class BacktestResult:
    """Kết quả tổng hợp của một lần chạy backtest."""

    trades: list[Trade]
    equity_curve: pd.Series          # index: ngày, value: giá trị danh mục
    initial_cash: float
    final_equity: float
    total_return_pct: float
    win_rate_pct: float
    sharpe_ratio: Optional[float]
    max_drawdown_pct: float
    n_trades: int
    fee_pct: float = 0.0


# ==============================================================================
# TIỆN ÍCH TẠO TÍN HIỆU (không bắt buộc dùng, chỉ là helper phổ biến)
# ==============================================================================

def make_crossover_signals(
    fast: pd.Series, slow: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Tạo cặp tín hiệu entry/exit dựa trên giao cắt giữa 2 đường chỉ báo
    (ví dụ: MA20 cắt lên/xuống EMA50).

    - entry = True tại ngày `fast` cắt LÊN `slow` (trước đó fast <= slow,
      nay fast > slow).
    - exit = True tại ngày `fast` cắt XUỐNG `slow` (trước đó fast >= slow,
      nay fast < slow).

    `fast` và `slow` cần có cùng độ dài / cùng index thứ tự với DataFrame
    gốc (ví dụ lấy trực tiếp từ core.indicators.calculate_ma/calculate_ema).
    """
    fast = fast.reset_index(drop=True)
    slow = slow.reset_index(drop=True)

    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)

    entry = (fast > slow) & (prev_fast <= prev_slow)
    exit_ = (fast < slow) & (prev_fast >= prev_slow)

    return entry.fillna(False), exit_.fillna(False)


# ==============================================================================
# BACKTEST ENGINE — chạy trên MỘT giai đoạn dữ liệu liên tục
# ==============================================================================

def _validate_df(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame thiếu các cột bắt buộc: {sorted(missing)}. "
            f"Cần đủ tối thiểu: {sorted(REQUIRED_COLUMNS)}."
        )


def run_backtest(
    df: pd.DataFrame,
    entry_signal_fn: Callable[[pd.DataFrame], pd.Series],
    exit_signal_fn: Callable[[pd.DataFrame], pd.Series],
    initial_cash: float = 100_000_000.0,
    fee_pct: float = 0.15,
    position_size_pct: float = 100.0,
) -> BacktestResult:
    """Chạy mô phỏng giao dịch trên danh mục ẢO theo tín hiệu cho trước.

    KHÔNG kết nối lệnh thật dưới bất kỳ hình thức nào — đây thuần túy là
    công cụ kiểm chứng trên dữ liệu lịch sử.

    Tham số:
        df: DataFrame OHLCV, tối thiểu cần cột 'date', 'open', 'close'.
        entry_signal_fn: hàm nhận df, trả về pd.Series bool cùng độ dài,
            True tại các ngày có tín hiệu MUA (đánh giá cuối ngày i, thực
            thi tại giá mở cửa ngày i+1).
        exit_signal_fn: tương tự, cho tín hiệu BÁN/đóng vị thế.
        initial_cash: vốn ảo ban đầu (VND).
        fee_pct: phí giao dịch tham khảo (%) áp dụng cho CẢ 2 CHIỀU
            mua/bán (phí môi giới + thuế bán, mặc định 0.15%).
        position_size_pct: % vốn khả dụng dùng cho mỗi lệnh mua (mặc định
            100% — dùng toàn bộ vốn hiện có).

    Trả về `BacktestResult`.
    """
    _validate_df(df)
    if not (0 < position_size_pct <= 100):
        raise ValueError("position_size_pct phải nằm trong khoảng (0, 100].")
    if len(df) < 2:
        raise ValueError("Cần tối thiểu 2 phiên dữ liệu để chạy backtest.")

    df_sorted = df.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)

    entry_signals = entry_signal_fn(df_sorted).reset_index(drop=True).fillna(False)
    exit_signals = exit_signal_fn(df_sorted).reset_index(drop=True).fillna(False)

    if len(entry_signals) != n or len(exit_signals) != n:
        raise ValueError(
            "entry_signal_fn/exit_signal_fn phải trả về Series có cùng độ "
            "dài với DataFrame đầu vào."
        )

    cash = initial_cash
    qty = 0
    in_position = False
    entry_price = 0.0
    entry_date: Optional[pd.Timestamp] = None
    entry_cost = 0.0
    entry_fee_amount = 0.0

    pending_action: Optional[str] = None  # "buy" | "sell" | None
    trades: list[Trade] = []
    equity_values: list[float] = []
    dates: list[pd.Timestamp] = []

    fee_rate = fee_pct / 100.0

    for i in range(n):
        today_date = df_sorted["date"].iloc[i]
        today_open = float(df_sorted["open"].iloc[i])
        today_close = float(df_sorted["close"].iloc[i])

        # --- Bước A: thực thi hành động còn treo từ tín hiệu hôm trước ---
        if pending_action == "buy" and not in_position:
            allocate_amount = cash * (position_size_pct / 100.0)
            max_qty = int(allocate_amount // (today_open * (1 + fee_rate)))
            if max_qty > 0:
                cost = max_qty * today_open
                fee_amount = cost * fee_rate
                cash -= (cost + fee_amount)
                qty = max_qty
                in_position = True
                entry_price = today_open
                entry_date = today_date
                entry_cost = cost
                entry_fee_amount = fee_amount
            pending_action = None

        elif pending_action == "sell" and in_position:
            proceeds = qty * today_open
            fee_amount = proceeds * fee_rate
            net_proceeds = proceeds - fee_amount
            pnl = net_proceeds - (entry_cost + entry_fee_amount)
            pnl_pct = (
                pnl / (entry_cost + entry_fee_amount) * 100.0
                if (entry_cost + entry_fee_amount) > 0
                else 0.0
            )
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=today_date,
                entry_price=entry_price,
                exit_price=today_open,
                qty=qty,
                entry_fee=entry_fee_amount,
                exit_fee=fee_amount,
                pnl=pnl,
                pnl_pct=pnl_pct,
            ))
            cash += net_proceeds
            qty = 0
            in_position = False
            entry_price = 0.0
            entry_date = None
            entry_cost = 0.0
            entry_fee_amount = 0.0
            pending_action = None

        # --- Bước B: định giá danh mục cuối ngày (mark-to-market) ---
        equity_today = cash + (qty * today_close if in_position else 0.0)
        equity_values.append(equity_today)
        dates.append(today_date)

        # --- Bước C: đánh giá tín hiệu cuối ngày để quyết định hành động
        #     cho phiên KẾ TIẾP (tránh lookahead bias) ---
        if i < n - 1:  # không còn "ngày mai" để thực thi nếu là phiên cuối
            if in_position:
                if bool(exit_signals.iloc[i]):
                    pending_action = "sell"
            else:
                if bool(entry_signals.iloc[i]):
                    pending_action = "buy"

    # --- Đóng cưỡng bức vị thế còn mở khi hết dữ liệu ---
    if in_position:
        last_close = float(df_sorted["close"].iloc[-1])
        last_date = df_sorted["date"].iloc[-1]
        proceeds = qty * last_close
        fee_amount = proceeds * fee_rate
        net_proceeds = proceeds - fee_amount
        pnl = net_proceeds - (entry_cost + entry_fee_amount)
        pnl_pct = (
            pnl / (entry_cost + entry_fee_amount) * 100.0
            if (entry_cost + entry_fee_amount) > 0
            else 0.0
        )
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=last_date,
            entry_price=entry_price,
            exit_price=last_close,
            qty=qty,
            entry_fee=entry_fee_amount,
            exit_fee=fee_amount,
            pnl=pnl,
            pnl_pct=pnl_pct,
            forced_close=True,
        ))
        cash += net_proceeds
        qty = 0
        in_position = False
        # Cập nhật lại giá trị equity của ngày cuối để phản ánh đúng phí
        # vừa phát sinh khi đóng cưỡng bức.
        equity_values[-1] = cash

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(dates), name="equity")

    # --- Tính các chỉ số hiệu suất ---
    final_equity = equity_values[-1]
    total_return_pct = (final_equity / initial_cash - 1.0) * 100.0

    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t.pnl > 0)
    win_rate_pct = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0

    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() != 0:
        sharpe_ratio = float(
            daily_returns.mean() / daily_returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe_ratio = None

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown_pct = float(abs(drawdown.min()) * 100.0) if len(drawdown) > 0 else 0.0

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        initial_cash=initial_cash,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        win_rate_pct=win_rate_pct,
        sharpe_ratio=sharpe_ratio,
        max_drawdown_pct=max_drawdown_pct,
        n_trades=n_trades,
        fee_pct=fee_pct,
    )


# ==============================================================================
# WALK-FORWARD: chia dữ liệu thành nhiều giai đoạn train/test theo thời gian
# ==============================================================================

def walk_forward_splits(
    df: pd.DataFrame, n_splits: int = 4
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Chia dữ liệu thành `n_splits` cặp (train, test) liên tiếp theo thời
    gian, dùng cửa sổ MỞ RỘNG DẦN (expanding window) cho phần train, để
    tránh đánh giá quá lạc quan (overfitting) trên một giai đoạn duy nhất.

    Cách chia: toàn bộ dữ liệu được cắt thành (n_splits + 1) đoạn bằng
    nhau theo thời gian. Đoạn đầu tiên luôn là train "khởi tạo"; từ đó,
    mỗi fold kế tiếp dùng toàn bộ dữ liệu TRƯỚC đoạn test hiện tại làm
    train (mở rộng dần), và đoạn kế tiếp làm test.
    """
    if n_splits < 1:
        raise ValueError("n_splits phải >= 1.")

    df_sorted = df.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)

    if n < (n_splits + 1) * 2:
        raise ValueError(
            f"Không đủ dữ liệu ({n} phiên) để chia thành {n_splits} fold "
            f"walk-forward một cách có ý nghĩa."
        )

    chunks = np.array_split(np.arange(n), n_splits + 1)

    results: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    train_indices = list(chunks[0])
    for i in range(1, n_splits + 1):
        test_indices = list(chunks[i])
        train_df = df_sorted.iloc[train_indices].reset_index(drop=True)
        test_df = df_sorted.iloc[test_indices].reset_index(drop=True)
        results.append((train_df, test_df))
        train_indices = train_indices + test_indices

    return results


def run_walk_forward_backtest(
    df: pd.DataFrame,
    entry_signal_fn: Callable[[pd.DataFrame], pd.Series],
    exit_signal_fn: Callable[[pd.DataFrame], pd.Series],
    n_splits: int = 4,
    **backtest_kwargs,
) -> list[BacktestResult]:
    """Chạy backtest lặp lại trên từng fold TEST của walk-forward split.

    Vì các quy tắc tín hiệu trong dự án này dựa trên luật cố định (không
    phải mô hình học máy cần huấn luyện tham số), phần `train_df` của mỗi
    fold hiện chưa được dùng để tối ưu tham số — nó được giữ lại trong
    `walk_forward_splits()` để dành cho việc mở rộng sau này (ví dụ tối ưu
    ngưỡng tín hiệu). Hàm này chỉ chạy backtest trên từng `test_df` để
    đánh giá tính ổn định của quy tắc qua nhiều giai đoạn thời gian khác
    nhau, tránh kết luận vội vàng từ một giai đoạn duy nhất.

    Trả về danh sách `BacktestResult`, mỗi phần tử ứng với một fold test.
    """
    splits = walk_forward_splits(df, n_splits=n_splits)
    results = []
    for _train_df, test_df in splits:
        if len(test_df) < 2:
            continue
        result = run_backtest(
            test_df, entry_signal_fn, exit_signal_fn, **backtest_kwargs
        )
        results.append(result)
    return results


# ==============================================================================
# BIỂU ĐỒ EQUITY CURVE
# ==============================================================================

def plot_equity_curve(
    result: BacktestResult,
    title: str = "Equity Curve",
    output_path: Optional[str] = None,
):
    """Vẽ biểu đồ equity curve (giá trị danh mục theo thời gian).

    Nếu `output_path` được truyền, lưu hình ảnh ra file tại đường dẫn đó.
    Luôn trả về đối tượng `Figure` của matplotlib để có thể nhúng trực
    tiếp vào dashboard (ví dụ qua `st.pyplot(fig)` trong Streamlit).
    """
    import matplotlib
    matplotlib.use("Agg")  # không cần môi trường đồ họa (headless-safe)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.equity_curve.index, result.equity_curve.values, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá trị danh mục (VND)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)

    return fig
