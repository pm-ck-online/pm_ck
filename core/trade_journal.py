"""
trade_journal.py
=================
[Bổ sung — Nhật ký giao dịch mua/bán mô phỏng]

Ghi nhận từng giao dịch MUA/BÁN MÔ PHỎNG do người dùng TỰ NHẬP (dựa trên
khuyến nghị hoặc phân tích riêng), kèm lý do và điểm tham chiếu kỹ thuật
(MA20/EMA50/EMA100/EMA200) tại thời điểm ra quyết định — để sau này đối
chiếu xem quyết định vào/ra lệnh có hợp lý hay không.

KHÔNG đặt lệnh giao dịch thật dưới bất kỳ hình thức nào — module này chỉ
ghi nhận và tính toán trên dữ liệu do người dùng tự nhập, phục vụ mục
đích theo dõi/rút kinh nghiệm.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Optional

VALID_REFERENCE_INDICATORS = {"MA20", "EMA50", "EMA100", "EMA200", "Khác"}


class InvalidTradeJournalError(ValueError):
    """Dữ liệu nhập vào nhật ký giao dịch không hợp lệ."""


def _validate_reference_indicator(value: str) -> None:
    if value not in VALID_REFERENCE_INDICATORS:
        raise InvalidTradeJournalError(
            f"reference_indicator '{value}' không hợp lệ. Cần một trong "
            f"{sorted(VALID_REFERENCE_INDICATORS)}."
        )


def create_trade_entry(
    symbol: str,
    qty: int,
    buy_price: float,
    buy_date: date,
    buy_reason: str = "",
    buy_reference_indicator: str = "Khác",
) -> dict:
    """Tạo một bản ghi giao dịch MUA mới (chưa bán) trong nhật ký.

    `buy_reference_indicator`: đường MA/EMA nào là căn cứ cho điểm vào
    lệnh này — dùng để sau này đối chiếu xem các quyết định mua dựa trên
    tín hiệu nào cho kết quả tốt/xấu.

    Trả về dict đại diện cho MỘT giao dịch, có `trade_id` DUY NHẤT để
    tham chiếu/đóng lệnh (ghi nhận bán) sau này.
    """
    if not symbol:
        raise InvalidTradeJournalError("symbol không được để trống.")
    if qty <= 0:
        raise InvalidTradeJournalError("qty phải > 0.")
    if buy_price <= 0:
        raise InvalidTradeJournalError("buy_price phải > 0.")
    _validate_reference_indicator(buy_reference_indicator)

    trade_id = f"{symbol}-{buy_date.isoformat()}-{uuid.uuid4().hex[:8]}"

    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "qty": qty,
        "buy_price": buy_price,
        "buy_date": buy_date.isoformat(),
        "buy_reason": buy_reason,
        "buy_reference_indicator": buy_reference_indicator,
        "sell_price": None,
        "sell_date": None,
        "sell_reason": "",
        "sell_reference_indicator": None,
        "pnl": None,
        "pnl_pct": None,
        "is_closed": False,
    }


def compute_pnl(buy_price: float, sell_price: float, qty: int) -> tuple[float, float]:
    """Tính lãi/lỗ tuyệt đối (VND) và % dựa trên giá mua/bán và khối lượng."""
    if buy_price <= 0:
        raise InvalidTradeJournalError("buy_price phải > 0 để tính PnL.")
    pnl = (sell_price - buy_price) * qty
    pnl_pct = (sell_price - buy_price) / buy_price * 100.0
    return pnl, pnl_pct


def close_trade_entry(
    entry: dict,
    sell_price: float,
    sell_date: date,
    sell_reason: str = "",
    sell_reference_indicator: str = "Khác",
) -> dict:
    """Đóng một vị thế đang mở (ghi nhận bán). Trả về BẢN SAO MỚI đã cập
    nhật đầy đủ thông tin bán + PnL — KHÔNG sửa trực tiếp `entry` gốc.
    """
    if entry.get("is_closed"):
        raise InvalidTradeJournalError(
            f"Giao dịch '{entry.get('trade_id')}' đã được đóng trước đó, "
            f"không thể đóng lại."
        )
    if sell_price <= 0:
        raise InvalidTradeJournalError("sell_price phải > 0.")
    _validate_reference_indicator(sell_reference_indicator)

    pnl, pnl_pct = compute_pnl(entry["buy_price"], sell_price, entry["qty"])

    updated = dict(entry)
    updated.update({
        "sell_price": sell_price,
        "sell_date": sell_date.isoformat(),
        "sell_reason": sell_reason,
        "sell_reference_indicator": sell_reference_indicator,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "is_closed": True,
    })
    return updated


def summarize_trades(trades: list[dict]) -> dict:
    """Tổng hợp thống kê nhanh trên danh sách giao dịch (mở + đã đóng).

    Trả về: {"total_pnl", "win_rate_pct", "n_closed", "n_open"}.
    """
    closed = [t for t in trades if t.get("is_closed")]
    open_trades = [t for t in trades if not t.get("is_closed")]

    total_pnl = sum(t["pnl"] for t in closed) if closed else 0.0
    n_wins = sum(1 for t in closed if t["pnl"] > 0)
    win_rate_pct = (n_wins / len(closed) * 100.0) if closed else 0.0

    return {
        "total_pnl": total_pnl,
        "win_rate_pct": win_rate_pct,
        "n_closed": len(closed),
        "n_open": len(open_trades),
    }
