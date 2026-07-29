"""
Unit test cho backtest/backtest_engine.py

Dùng kịch bản dữ liệu giá THỦ CÔNG (đã tính tay trước kết quả kỳ vọng) để
kiểm chứng chính xác logic mua/bán, tính phí, equity curve, và các chỉ số
hiệu suất — không phụ thuộc dữ liệu ngẫu nhiên khó kiểm chứng.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from backtest.backtest_engine import (
    make_crossover_signals,
    plot_equity_curve,
    run_backtest,
    run_walk_forward_backtest,
    walk_forward_splits,
)


def _make_df(prices: list[float], start_date: str = "2024-01-01") -> pd.DataFrame:
    """Tạo DataFrame OHLCV đơn giản: open = close = giá truyền vào (mỗi
    ngày một mức giá cố định, không dao động trong ngày) — giúp tính tay
    kết quả kỳ vọng dễ dàng và chính xác.
    """
    n = len(prices)
    dates = pd.bdate_range(start=start_date, periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1000] * n,
    })


def _signal_fn_from_list(flags: list[bool]):
    def _fn(df: pd.DataFrame) -> pd.Series:
        return pd.Series(flags)
    return _fn


# ==============================================================================
# Test: kịch bản thủ công — 1 lệnh có lãi/lỗ, không phí (fee_pct=0)
# ==============================================================================

class TestManualScenarioNoFee:
    """Kịch bản 5 phiên, giá: 100, 110, 90, 95, 130.

    entry_signal = [True, False, False, False, False]  -> quyết định mua
        cuối ngày 0, thực thi tại giá mở cửa ngày 1 (=110).
    exit_signal  = [False, False, True, False, False]  -> quyết định bán
        cuối ngày 2, thực thi tại giá mở cửa ngày 3 (=95).

    Với vốn ban đầu 1,000,000 và fee=0:
        qty = floor(1,000,000 / 110) = 9090
        entry_cost = 9090 * 110 = 999,900 -> cash còn lại = 100
        Ngày 1 (mua xong): equity = 100 + 9090*110(close) = 1,000,000
        Ngày 2: equity = 100 + 9090*90 = 818,200
        Ngày 3 (bán tại open=95): proceeds = 9090*95 = 863,550
            cash = 100 + 863,550 = 863,650
            pnl = 863,550 - 999,900 = -136,350
        Ngày 4: equity = 863,650 (giữ nguyên, không còn vị thế)
    """

    @pytest.fixture
    def result(self):
        df = _make_df([100, 110, 90, 95, 130])
        entry_fn = _signal_fn_from_list([True, False, False, False, False])
        exit_fn = _signal_fn_from_list([False, False, True, False, False])
        return run_backtest(
            df, entry_fn, exit_fn,
            initial_cash=1_000_000.0, fee_pct=0.0, position_size_pct=100.0,
        )

    def test_n_trades(self, result):
        assert result.n_trades == 1

    def test_trade_details(self, result):
        trade = result.trades[0]
        assert trade.qty == 9090
        assert trade.entry_price == pytest.approx(110)
        assert trade.exit_price == pytest.approx(95)
        assert trade.pnl == pytest.approx(-136_350, abs=1)
        assert trade.forced_close is False

    def test_equity_curve_values(self, result):
        values = result.equity_curve.values
        assert values[0] == pytest.approx(1_000_000, abs=1)   # ngày 0: chưa mua
        assert values[1] == pytest.approx(1_000_000, abs=1)   # ngày 1: vừa mua, giá không đổi
        assert values[2] == pytest.approx(818_200, abs=1)     # ngày 2: giá giảm còn 90
        assert values[3] == pytest.approx(863_650, abs=1)     # ngày 3: vừa bán
        assert values[4] == pytest.approx(863_650, abs=1)     # ngày 4: giữ nguyên

    def test_final_equity_and_total_return(self, result):
        assert result.final_equity == pytest.approx(863_650, abs=1)
        expected_return_pct = (863_650 / 1_000_000 - 1) * 100
        assert result.total_return_pct == pytest.approx(expected_return_pct, abs=0.01)

    def test_win_rate_zero_for_losing_trade(self, result):
        assert result.win_rate_pct == pytest.approx(0.0)

    def test_max_drawdown(self, result):
        # Đỉnh equity = 1,000,000 (ngày 0-1), đáy sau đó = 818,200 (ngày 2)
        expected_dd = abs((818_200 - 1_000_000) / 1_000_000) * 100
        assert result.max_drawdown_pct == pytest.approx(expected_dd, abs=0.01)

    def test_sharpe_ratio_is_computed(self, result):
        assert result.sharpe_ratio is not None
        assert isinstance(result.sharpe_ratio, float)


# ==============================================================================
# Test: có phí giao dịch (fee_pct > 0) — kiểm tra phí trừ đúng cả 2 chiều
# ==============================================================================

class TestFeeCalculation:
    def test_fee_reduces_pnl_correctly(self):
        df = _make_df([100, 100, 100, 100])
        entry_fn = _signal_fn_from_list([True, False, False, False])
        exit_fn = _signal_fn_from_list([False, True, False, False])

        result = run_backtest(
            df, entry_fn, exit_fn,
            initial_cash=1_000_000.0, fee_pct=0.5, position_size_pct=100.0,
        )

        trade = result.trades[0]
        # qty = floor(1,000,000 / (100 * 1.005)) = floor(9950.24...) = 9950
        expected_qty = int(1_000_000 // (100 * 1.005))
        assert trade.qty == expected_qty

        entry_cost = expected_qty * 100
        entry_fee = entry_cost * 0.005
        exit_proceeds = expected_qty * 100
        exit_fee = exit_proceeds * 0.005
        expected_pnl = (exit_proceeds - exit_fee) - (entry_cost + entry_fee)

        # Giá không đổi (100 -> 100) nhưng vẫn LỖ đúng bằng tổng phí 2 chiều
        assert trade.pnl == pytest.approx(expected_pnl, abs=1)
        assert trade.pnl < 0  # có phí thì dù giá đứng yên vẫn lỗ nhẹ


# ==============================================================================
# Test: không có tín hiệu nào -> không giao dịch
# ==============================================================================

class TestNoSignals:
    def test_no_trades_when_no_signals_fire(self):
        df = _make_df([100, 101, 99, 102, 100])
        entry_fn = _signal_fn_from_list([False] * 5)
        exit_fn = _signal_fn_from_list([False] * 5)

        result = run_backtest(df, entry_fn, exit_fn, initial_cash=1_000_000.0)

        assert result.n_trades == 0
        assert result.final_equity == pytest.approx(1_000_000.0)
        assert result.total_return_pct == pytest.approx(0.0)
        assert result.win_rate_pct == pytest.approx(0.0)


# ==============================================================================
# Test: đóng cưỡng bức vị thế còn mở khi hết dữ liệu
# ==============================================================================

class TestForcedClose:
    def test_forces_close_open_position_at_end(self):
        df = _make_df([100, 110, 120, 130])
        entry_fn = _signal_fn_from_list([True, False, False, False])
        exit_fn = _signal_fn_from_list([False, False, False, False])  # không bao giờ bán

        result = run_backtest(df, entry_fn, exit_fn, initial_cash=1_000_000.0, fee_pct=0.0)

        assert result.n_trades == 1
        assert result.trades[0].forced_close is True
        assert result.trades[0].exit_price == pytest.approx(130)  # giá đóng cửa phiên cuối


# ==============================================================================
# Test: make_crossover_signals
# ==============================================================================

class TestCrossoverSignals:
    def test_entry_on_upward_cross(self):
        fast = pd.Series([1, 2, 5, 4, 3])
        slow = pd.Series([3, 3, 3, 3, 3])
        entry, exit_ = make_crossover_signals(fast, slow)

        # fast cắt lên slow tại index 2 (từ 2<=3 sang 5>3)
        assert entry.tolist() == [False, False, True, False, False]
        # fast cắt xuống slow tại index 4 (từ 4>=3 sang 3... bằng, không tính là cắt xuống nghiêm ngặt)
        # kiểm tra tại điểm rõ ràng cắt xuống thay vì suy diễn from trên
        fast2 = pd.Series([5, 5, 5, 2, 2])
        slow2 = pd.Series([3, 3, 3, 3, 3])
        _, exit2 = make_crossover_signals(fast2, slow2)
        assert exit2.tolist() == [False, False, False, True, False]


# ==============================================================================
# Test: walk_forward_splits
# ==============================================================================

class TestWalkForwardSplits:
    def test_correct_number_of_folds(self):
        df = _make_df(list(range(100)))
        splits = walk_forward_splits(df, n_splits=4)
        assert len(splits) == 4

    def test_train_expands_across_folds(self):
        df = _make_df(list(range(100)))
        splits = walk_forward_splits(df, n_splits=4)
        train_sizes = [len(train_df) for train_df, _ in splits]
        # Train phải mở rộng dần (không giảm) qua từng fold
        assert train_sizes == sorted(train_sizes)

    def test_raises_on_insufficient_data(self):
        df = _make_df(list(range(5)))
        with pytest.raises(ValueError):
            walk_forward_splits(df, n_splits=4)


# ==============================================================================
# Test: run_walk_forward_backtest
# ==============================================================================

class TestRunWalkForwardBacktest:
    def test_returns_result_per_fold(self):
        df = _make_df([100 + (i % 10) for i in range(120)])

        def entry_fn(d: pd.DataFrame) -> pd.Series:
            return pd.Series([i % 15 == 0 for i in range(len(d))])

        def exit_fn(d: pd.DataFrame) -> pd.Series:
            return pd.Series([i % 15 == 7 for i in range(len(d))])

        results = run_walk_forward_backtest(
            df, entry_fn, exit_fn, n_splits=3, initial_cash=1_000_000.0
        )
        assert len(results) == 3
        for r in results:
            assert r.final_equity > 0


# ==============================================================================
# Test: plot_equity_curve
# ==============================================================================

class TestPlotEquityCurve:
    def test_returns_figure_and_saves_file(self, tmp_path):
        df = _make_df([100, 105, 110, 108, 115])
        entry_fn = _signal_fn_from_list([True, False, False, False, False])
        exit_fn = _signal_fn_from_list([False, False, True, False, False])
        result = run_backtest(df, entry_fn, exit_fn, initial_cash=1_000_000.0)

        output_file = tmp_path / "equity_curve.png"
        fig = plot_equity_curve(result, output_path=str(output_file))

        assert fig is not None
        assert output_file.exists()
        assert output_file.stat().st_size > 0


# ==============================================================================
# Test: xác thực đầu vào
# ==============================================================================

class TestInputValidation:
    def test_missing_required_columns_raises(self):
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5)})
        with pytest.raises(ValueError):
            run_backtest(df, _signal_fn_from_list([False] * 5), _signal_fn_from_list([False] * 5))

    def test_invalid_position_size_pct_raises(self):
        df = _make_df([100, 101, 102])
        with pytest.raises(ValueError):
            run_backtest(
                df, _signal_fn_from_list([False] * 3), _signal_fn_from_list([False] * 3),
                position_size_pct=150,
            )

    def test_mismatched_signal_length_raises(self):
        df = _make_df([100, 101, 102, 103])

        def bad_entry_fn(_df):
            return pd.Series([True, False])  # sai độ dài

        with pytest.raises(ValueError):
            run_backtest(df, bad_entry_fn, _signal_fn_from_list([False] * 4))

    def test_too_few_rows_raises(self):
        df = _make_df([100])
        with pytest.raises(ValueError):
            run_backtest(df, _signal_fn_from_list([False]), _signal_fn_from_list([False]))
