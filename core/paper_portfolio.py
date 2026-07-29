"""
paper_portfolio.py
====================
[Giai đoạn 4 — Danh mục mô phỏng]

Danh mục đầu tư MÔ PHỎNG (paper trading) — KHÔNG kết nối tài khoản thật,
KHÔNG tự động đặt lệnh. Người dùng tự nhập lệnh mô phỏng dựa trên khuyến
nghị từ capital_allocator, module này chỉ ghi nhận và tính toán.

Khác với các module trước (thuần hàm/stateless), module này giữ TRẠNG
THÁI (vốn, vị thế, lịch sử giao dịch) qua nhiều lần gọi trong một phiên
chạy chương trình — nên được thiết kế dưới dạng class `PaperPortfolio`.
Việc lưu trữ bền vững (ghi ra đĩa) do core/storage.py đảm nhiệm ở bước
khác, module này chỉ quản lý logic trong bộ nhớ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class InvalidTradeError(ValueError):
    """Lệnh mô phỏng không hợp lệ (side sai, qty/giá <= 0...)."""


class InsufficientCashError(ValueError):
    """Không đủ tiền mặt trong danh mục để thực hiện lệnh mua."""


class InsufficientPositionError(ValueError):
    """Không đủ khối lượng đang nắm giữ để thực hiện lệnh bán."""


# ==============================================================================
# CẤU TRÚC DỮ LIỆU
# ==============================================================================

@dataclass
class Position:
    """Một vị thế đang nắm giữ trong danh mục mô phỏng."""

    symbol: str
    qty: int
    avg_cost: float   # giá vốn bình quân (đã gộp phí)
    sector: Optional[str] = None


@dataclass
class TradeRecord:
    """Một bản ghi lệnh mô phỏng — lưu lại đầy đủ để phục vụ rút kinh
    nghiệm sau này, đặc biệt là `entry_range_at_signal` để đối chiếu xem
    người dùng có vào lệnh đúng vùng entry đã được khuyến nghị hay không.
    """

    timestamp: datetime
    symbol: str
    side: str          # "buy" | "sell"
    qty: int
    price: float
    fee: float
    entry_range_at_signal: Optional[dict] = None  # {"low":..., "high":...}
    realized_pnl: Optional[float] = None  # chỉ có giá trị với lệnh "sell"
    notes: str = ""


# ==============================================================================
# DANH MỤC MÔ PHỎNG
# ==============================================================================

class PaperPortfolio:
    """Danh mục đầu tư mô phỏng — quản lý vốn ảo, vị thế, và lịch sử giao
    dịch mô phỏng. KHÔNG kết nối tài khoản thật.
    """

    def __init__(self, initial_cash: float):
        if initial_cash <= 0:
            raise InvalidTradeError("initial_cash phải > 0.")
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.trade_history: list[TradeRecord] = []

    # --------------------------------------------------------------------
    # Ghi nhận lệnh mô phỏng
    # --------------------------------------------------------------------
    def record_trade(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        entry_range_at_signal: Optional[dict] = None,
        fee_pct: float = 0.0,
        sector: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        notes: str = "",
    ) -> TradeRecord:
        """Ghi nhận MỘT lệnh mua/bán MÔ PHỎNG theo giá thị trường thực tế
        tại thời điểm người dùng tự nhập lệnh (không tự động đặt lệnh).

        Tham số:
            symbol: mã cổ phiếu.
            side: "buy" hoặc "sell".
            qty: khối lượng (phải > 0).
            price: giá khớp lệnh mô phỏng (phải > 0).
            entry_range_at_signal: vùng giá entry đã được
                `capital_allocator` khuyến nghị TẠI THỜI ĐIỂM ra quyết
                định này — lưu lại để sau này đối chiếu xem người dùng có
                vào đúng vùng entry hay không.
            fee_pct: phí giao dịch tham khảo (%) áp dụng cho lệnh này.
            sector: ngành của mã (dùng cho việc so sánh tỷ trọng theo
                ngành sau này nếu cần).
            timestamp: thời điểm ghi nhận (mặc định là hiện tại).
            notes: ghi chú tự do.

        Trả về `TradeRecord` vừa được thêm vào lịch sử.
        """
        if side not in {"buy", "sell"}:
            raise InvalidTradeError(f"side phải là 'buy' hoặc 'sell', nhận '{side}'.")
        if qty <= 0:
            raise InvalidTradeError("qty phải > 0.")
        if price <= 0:
            raise InvalidTradeError("price phải > 0.")

        timestamp = timestamp or datetime.now()
        realized_pnl: Optional[float] = None

        if side == "buy":
            cost = qty * price
            fee = cost * (fee_pct / 100.0)
            total_cost = cost + fee

            if total_cost > self.cash:
                raise InsufficientCashError(
                    f"Không đủ tiền mặt: cần {total_cost:,.2f} nhưng chỉ còn "
                    f"{self.cash:,.2f}."
                )

            self.cash -= total_cost

            if symbol in self.positions:
                existing = self.positions[symbol]
                new_qty = existing.qty + qty
                # Giá vốn bình quân mới, gộp cả phí giao dịch của lần mua này.
                new_avg_cost = (
                    existing.avg_cost * existing.qty + cost + fee
                ) / new_qty
                existing.qty = new_qty
                existing.avg_cost = new_avg_cost
                if sector and not existing.sector:
                    existing.sector = sector
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    avg_cost=(cost + fee) / qty,
                    sector=sector,
                )

        else:  # side == "sell"
            existing = self.positions.get(symbol)
            if existing is None or existing.qty < qty:
                held = existing.qty if existing else 0
                raise InsufficientPositionError(
                    f"Không đủ khối lượng '{symbol}' để bán: đang nắm giữ "
                    f"{held}, yêu cầu bán {qty}."
                )

            proceeds = qty * price
            fee = proceeds * (fee_pct / 100.0)
            net_proceeds = proceeds - fee
            self.cash += net_proceeds

            realized_pnl = net_proceeds - existing.avg_cost * qty

            existing.qty -= qty
            if existing.qty == 0:
                del self.positions[symbol]

        record = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            fee=(qty * price * (fee_pct / 100.0)),
            entry_range_at_signal=entry_range_at_signal,
            realized_pnl=realized_pnl,
            notes=notes,
        )
        self.trade_history.append(record)
        return record

    # --------------------------------------------------------------------
    # Snapshot danh mục — vị thế, PnL, tỷ trọng
    # --------------------------------------------------------------------
    def get_portfolio_snapshot(self, current_prices: dict[str, float]) -> dict:
        """Tính snapshot hiện tại của danh mục: vị thế, PnL chưa thực hiện
        (unrealized), NAV, và tỷ trọng thực tế của từng vị thế.

        `current_prices`: dict {symbol: giá hiện tại} — nên lấy từ
        `data_collector.get_realtime_price()` cho từng mã đang nắm giữ.
        Nếu thiếu giá của một mã đang nắm giữ, dùng tạm giá vốn bình quân
        để định giá (mark-to-market) và ghi rõ cảnh báo trong kết quả.
        """
        positions_snapshot: list[dict] = []
        total_market_value = 0.0
        total_unrealized_pnl = 0.0
        warnings: list[str] = []

        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol)
            if price is None:
                price = pos.avg_cost
                warnings.append(
                    f"Thiếu giá hiện tại cho '{symbol}' -> tạm dùng giá vốn "
                    f"bình quân ({pos.avg_cost:,.2f}) để định giá."
                )

            market_value = pos.qty * price
            unrealized_pnl = market_value - pos.avg_cost * pos.qty
            unrealized_pnl_pct = (
                unrealized_pnl / (pos.avg_cost * pos.qty) * 100.0
                if pos.avg_cost * pos.qty > 0
                else 0.0
            )

            positions_snapshot.append({
                "symbol": symbol,
                "qty": pos.qty,
                "avg_cost": pos.avg_cost,
                "current_price": price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "sector": pos.sector,
            })

            total_market_value += market_value
            total_unrealized_pnl += unrealized_pnl

        nav = self.cash + total_market_value

        for p in positions_snapshot:
            p["weight_pct"] = (p["market_value"] / nav * 100.0) if nav > 0 else 0.0

        total_stock_weight_pct = (total_market_value / nav * 100.0) if nav > 0 else 0.0
        total_realized_pnl = sum(
            t.realized_pnl for t in self.trade_history if t.realized_pnl is not None
        )

        return {
            "cash": self.cash,
            "nav": nav,
            "total_market_value": total_market_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_realized_pnl": total_realized_pnl,
            "total_return_pct": (
                (nav - self.initial_cash) / self.initial_cash * 100.0
                if self.initial_cash > 0
                else 0.0
            ),
            "total_stock_weight_pct": total_stock_weight_pct,
            "positions": positions_snapshot,
            "warnings": warnings,
        }

    # --------------------------------------------------------------------
    # So sánh tỷ trọng thực tế với khuyến nghị (capital_allocator)
    # --------------------------------------------------------------------
    def compare_to_target_allocation(
        self,
        current_prices: dict[str, float],
        target_pct: float,
        deviation_threshold_pct: float = 10.0,
    ) -> dict:
        """So sánh tỷ trọng cổ phiếu THỰC TẾ trong danh mục với tỷ trọng
        KHUYẾN NGHỊ từ `capital_allocator.get_allocation_recommendation()`,
        cảnh báo khi độ lệch vượt quá `deviation_threshold_pct`.

        Trả về dict: {"actual_pct", "target_pct", "deviation_pct",
        "exceeds_threshold": bool, "note": str}.
        """
        snapshot = self.get_portfolio_snapshot(current_prices)
        actual_pct = snapshot["total_stock_weight_pct"]
        deviation_pct = actual_pct - target_pct
        exceeds_threshold = abs(deviation_pct) > deviation_threshold_pct

        if exceeds_threshold:
            direction = "CAO HƠN" if deviation_pct > 0 else "THẤP HƠN"
            note = (
                f"Tỷ trọng cổ phiếu thực tế ({actual_pct:.1f}%) đang {direction} "
                f"tỷ trọng khuyến nghị ({target_pct:.1f}%) tới "
                f"{abs(deviation_pct):.1f} điểm % -> vượt ngưỡng cảnh báo "
                f"{deviation_threshold_pct:.1f}%."
            )
        else:
            note = "Tỷ trọng thực tế đang nằm trong ngưỡng chấp nhận được so với khuyến nghị."

        return {
            "actual_pct": actual_pct,
            "target_pct": target_pct,
            "deviation_pct": deviation_pct,
            "exceeds_threshold": exceeds_threshold,
            "note": note,
        }

    # --------------------------------------------------------------------
    # Tiện ích
    # --------------------------------------------------------------------
    def get_trade_history(self, symbol: Optional[str] = None) -> list[TradeRecord]:
        """Trả về lịch sử giao dịch, lọc theo mã nếu có truyền `symbol`."""
        if symbol is None:
            return list(self.trade_history)
        return [t for t in self.trade_history if t.symbol == symbol]


# ==============================================================================
# HÀM KHỞI TẠO (theo đúng tên hàm đề xuất trong yêu cầu dự án)
# ==============================================================================

def create_portfolio(initial_cash: float) -> PaperPortfolio:
    """Khởi tạo một danh mục đầu tư MÔ PHỎNG mới với số vốn ảo ban đầu."""
    return PaperPortfolio(initial_cash)
