"""
Unit test cho core/trade_journal.py
"""

from __future__ import annotations

from datetime import date

import pytest

from core.trade_journal import (
    InvalidTradeJournalError,
    close_trade_entry,
    compute_pnl,
    create_trade_entry,
    summarize_trades,
)


# ==============================================================================
# Test: create_trade_entry
# ==============================================================================

class TestCreateTradeEntry:
    def test_creates_open_trade_with_expected_fields(self):
        entry = create_trade_entry(
            symbol="HPG", qty=100, buy_price=20.5, buy_date=date(2026, 1, 15),
            buy_reason="Breakout khỏi vùng tích lũy", buy_reference_indicator="EMA50",
        )
        assert entry["symbol"] == "HPG"
        assert entry["qty"] == 100
        assert entry["buy_price"] == 20.5
        assert entry["buy_date"] == "2026-01-15"
        assert entry["buy_reference_indicator"] == "EMA50"
        assert entry["is_closed"] is False
        assert entry["sell_price"] is None
        assert entry["pnl"] is None

    def test_trade_id_is_unique_across_calls(self):
        e1 = create_trade_entry("HPG", 100, 20.5, date(2026, 1, 15))
        e2 = create_trade_entry("HPG", 100, 20.5, date(2026, 1, 15))
        assert e1["trade_id"] != e2["trade_id"]

    def test_raises_for_empty_symbol(self):
        with pytest.raises(InvalidTradeJournalError):
            create_trade_entry("", 100, 20.5, date(2026, 1, 15))

    def test_raises_for_non_positive_qty(self):
        with pytest.raises(InvalidTradeJournalError):
            create_trade_entry("HPG", 0, 20.5, date(2026, 1, 15))

    def test_raises_for_non_positive_price(self):
        with pytest.raises(InvalidTradeJournalError):
            create_trade_entry("HPG", 100, 0, date(2026, 1, 15))

    def test_raises_for_invalid_reference_indicator(self):
        with pytest.raises(InvalidTradeJournalError):
            create_trade_entry(
                "HPG", 100, 20.5, date(2026, 1, 15),
                buy_reference_indicator="RSI",  # không nằm trong danh sách hợp lệ
            )


# ==============================================================================
# Test: compute_pnl
# ==============================================================================

class TestComputePnl:
    def test_profit_calculation(self):
        pnl, pnl_pct = compute_pnl(buy_price=20.0, sell_price=22.0, qty=100)
        assert pnl == pytest.approx(200.0)
        assert pnl_pct == pytest.approx(10.0)

    def test_loss_calculation(self):
        pnl, pnl_pct = compute_pnl(buy_price=20.0, sell_price=18.0, qty=100)
        assert pnl == pytest.approx(-200.0)
        assert pnl_pct == pytest.approx(-10.0)

    def test_raises_for_non_positive_buy_price(self):
        with pytest.raises(InvalidTradeJournalError):
            compute_pnl(buy_price=0, sell_price=10, qty=100)


# ==============================================================================
# Test: close_trade_entry
# ==============================================================================

class TestCloseTradeEntry:
    def test_closes_open_trade_correctly(self):
        entry = create_trade_entry("HPG", 100, 20.0, date(2026, 1, 15))
        closed = close_trade_entry(
            entry, sell_price=22.0, sell_date=date(2026, 2, 1),
            sell_reason="Chạm kháng cự", sell_reference_indicator="EMA200",
        )
        assert closed["is_closed"] is True
        assert closed["sell_price"] == 22.0
        assert closed["sell_date"] == "2026-02-01"
        assert closed["sell_reference_indicator"] == "EMA200"
        assert closed["pnl"] == pytest.approx(200.0)
        assert closed["pnl_pct"] == pytest.approx(10.0)

    def test_does_not_mutate_original_entry(self):
        entry = create_trade_entry("HPG", 100, 20.0, date(2026, 1, 15))
        close_trade_entry(entry, sell_price=22.0, sell_date=date(2026, 2, 1))
        # entry gốc KHÔNG được thay đổi
        assert entry["is_closed"] is False
        assert entry["sell_price"] is None

    def test_raises_when_closing_already_closed_trade(self):
        entry = create_trade_entry("HPG", 100, 20.0, date(2026, 1, 15))
        closed = close_trade_entry(entry, sell_price=22.0, sell_date=date(2026, 2, 1))
        with pytest.raises(InvalidTradeJournalError):
            close_trade_entry(closed, sell_price=23.0, sell_date=date(2026, 2, 5))

    def test_raises_for_non_positive_sell_price(self):
        entry = create_trade_entry("HPG", 100, 20.0, date(2026, 1, 15))
        with pytest.raises(InvalidTradeJournalError):
            close_trade_entry(entry, sell_price=0, sell_date=date(2026, 2, 1))

    def test_raises_for_invalid_sell_reference_indicator(self):
        entry = create_trade_entry("HPG", 100, 20.0, date(2026, 1, 15))
        with pytest.raises(InvalidTradeJournalError):
            close_trade_entry(
                entry, sell_price=22.0, sell_date=date(2026, 2, 1),
                sell_reference_indicator="MACD",
            )


# ==============================================================================
# Test: summarize_trades
# ==============================================================================

class TestSummarizeTrades:
    def test_summary_with_mixed_open_and_closed_trades(self):
        open_trade = create_trade_entry("VNM", 50, 58.0, date(2026, 1, 10))

        win_trade = create_trade_entry("HPG", 100, 20.0, date(2026, 1, 15))
        win_trade = close_trade_entry(win_trade, sell_price=22.0, sell_date=date(2026, 2, 1))

        loss_trade = create_trade_entry("FPT", 50, 65.0, date(2026, 1, 20))
        loss_trade = close_trade_entry(loss_trade, sell_price=60.0, sell_date=date(2026, 2, 5))

        summary = summarize_trades([open_trade, win_trade, loss_trade])

        assert summary["n_open"] == 1
        assert summary["n_closed"] == 2
        assert summary["win_rate_pct"] == pytest.approx(50.0)
        expected_total_pnl = (22.0 - 20.0) * 100 + (60.0 - 65.0) * 50
        assert summary["total_pnl"] == pytest.approx(expected_total_pnl)

    def test_summary_with_no_trades(self):
        summary = summarize_trades([])
        assert summary["n_open"] == 0
        assert summary["n_closed"] == 0
        assert summary["win_rate_pct"] == 0.0
        assert summary["total_pnl"] == 0.0
