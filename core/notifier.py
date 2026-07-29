"""
notifier.py
============
[Giai đoạn 4 — Cảnh báo]

Gửi cảnh báo qua Telegram. Thiết kế theo ADAPTER PATTERN (giống
`core/data_collector.py`): lớp `TelegramClient` là interface trừu tượng,
`RealTelegramClient` gọi Bot API thật, còn khi TEST chỉ cần truyền vào một
client giả lập — KHÔNG bao giờ gọi Telegram thật khi chạy unit test.

GỬI CẢNH BÁO KHI:
    1. Giá chạm ngưỡng theo dõi (watchlist alert).
    2. market_regime_detector báo thay đổi giai đoạn thị trường (kèm lý
       do từ trường "reasoning").
    3. pattern_detector phát hiện mô hình thu hẹp biên độ mới với
       confidence cao trên một mã đang theo dõi.
    4. paper_portfolio vượt ngưỡng lãi/lỗ đã đặt trước.
    5. data_collector cảnh báo dữ liệu không cập nhật quá X phút.

BẢO MẬT:
    - Token bot đọc từ BIẾN MÔI TRƯỜNG, KHÔNG hardcode trong code.
    - chat_id/user_id gửi LỆNH ĐIỀU KHIỂN (vd. /pause, /status) PHẢI nằm
      trong whitelist cấu hình được — không tin bất kỳ người gửi lạ nào.
      Whitelist CHỈ áp dụng cho lệnh điều khiển ĐI VÀO hệ thống; việc HỆ
      THỐNG chủ động gửi cảnh báo ra ngoài không cần kiểm tra whitelist
      (vì chat_id đích do chính cấu hình của người dùng chỉ định).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class NotifierError(Exception):
    """Lỗi phát sinh khi gửi thông báo (thiếu token, lỗi mạng...)."""


class UnauthorizedCommandError(Exception):
    """chat_id gửi lệnh điều khiển KHÔNG nằm trong whitelist."""


# ==============================================================================
# INTERFACE CLIENT TELEGRAM (Adapter Pattern)
# ==============================================================================

class TelegramClient(ABC):
    """Interface trừu tượng cho việc gửi tin nhắn Telegram.

    `RealTelegramClient` cài đặt gọi Bot API thật; khi test, truyền vào
    một client giả lập (mock) kế thừa lớp này để không gọi mạng thật.
    """

    @abstractmethod
    def send_message(self, chat_id: str, text: str) -> bool:
        """Gửi tin nhắn `text` tới `chat_id`. Trả về True nếu thành công."""


class RealTelegramClient(TelegramClient):
    """Client thật, gọi Telegram Bot API qua HTTP.

    Token đọc từ BIẾN MÔI TRƯỜNG có tên `token_env_var` (KHÔNG hardcode
    token trong code hay truyền trực tiếp giá trị token vào constructor).
    """

    def __init__(self, token_env_var: str = "TELEGRAM_BOT_TOKEN"):
        self.token_env_var = token_env_var

    def _get_token(self) -> str:
        token = os.environ.get(self.token_env_var)
        if not token:
            raise NotifierError(
                f"Không tìm thấy TELEGRAM BOT TOKEN trong biến môi trường "
                f"'{self.token_env_var}'. Hãy đặt biến môi trường này trước "
                f"khi gửi thông báo (KHÔNG hardcode token trong code)."
            )
        return token

    def send_message(self, chat_id: str, text: str) -> bool:
        token = self._get_token()  # kiểm tra token TRƯỚC, không phụ thuộc requests nếu thiếu token

        import requests  # import cục bộ: chỉ cần khi thực sự gửi tin nhắn thật

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        response = requests.post(
            url, data={"chat_id": chat_id, "text": text}, timeout=10
        )
        return response.status_code == 200


# ==============================================================================
# NOTIFIER — điều phối chính
# ==============================================================================

class Notifier:
    """Điều phối việc gửi cảnh báo qua Telegram cho toàn hệ thống.

    Đây là điểm truy cập DUY NHẤT mà các module khác nên dùng để gửi
    thông báo — không tự gọi thẳng `TelegramClient`.
    """

    def __init__(
        self,
        client: TelegramClient,
        whitelist_chat_ids: Optional[list[str]] = None,
        default_chat_id: Optional[str] = None,
    ):
        self.client = client
        self.whitelist_chat_ids = set(whitelist_chat_ids or [])
        self.default_chat_id = default_chat_id
        self.sent_log: list[dict] = []  # lưu lại lịch sử đã gửi, phục vụ audit/test

    # --------------------------------------------------------------------
    # Gửi thông báo (dùng chung cho mọi loại cảnh báo)
    # --------------------------------------------------------------------
    def _send(self, chat_id: Optional[str], text: str, category: str) -> bool:
        target_chat_id = chat_id or self.default_chat_id
        if not target_chat_id:
            raise NotifierError(
                "Không xác định được chat_id đích để gửi thông báo (không "
                "truyền chat_id và cũng không có default_chat_id)."
            )

        success = self.client.send_message(target_chat_id, text)
        self.sent_log.append({
            "chat_id": target_chat_id,
            "category": category,
            "text": text,
            "success": success,
        })
        return success

    # --------------------------------------------------------------------
    # 1. Cảnh báo giá chạm ngưỡng theo dõi (watchlist)
    # --------------------------------------------------------------------
    def send_watchlist_alert(
        self,
        symbol: str,
        current_price: float,
        threshold_price: float,
        direction: str,  # "above" | "below"
        chat_id: Optional[str] = None,
    ) -> bool:
        if direction not in {"above", "below"}:
            raise ValueError("direction phải là 'above' hoặc 'below'.")

        arrow = "vượt LÊN TRÊN" if direction == "above" else "giảm XUỐNG DƯỚI"
        text = (
            f"🔔 [Watchlist] {symbol}: giá hiện tại {current_price:,.2f} đã "
            f"{arrow} ngưỡng theo dõi {threshold_price:,.2f}."
        )
        return self._send(chat_id, text, category="watchlist")

    # --------------------------------------------------------------------
    # 2. Cảnh báo đổi giai đoạn thị trường (market_regime_detector)
    # --------------------------------------------------------------------
    def send_regime_change_alert(
        self,
        sector_or_market: str,
        old_regime: Optional[str],
        new_regime: str,
        reasoning: list[str],
        chat_id: Optional[str] = None,
    ) -> bool:
        reasoning_text = "\n".join(f"  - {r}" for r in reasoning)
        text = (
            f"📊 [Đổi giai đoạn thị trường] {sector_or_market}: "
            f"{old_regime or 'chưa xác định'} -> {new_regime}\n"
            f"Lý do:\n{reasoning_text}"
        )
        return self._send(chat_id, text, category="regime_change")

    # --------------------------------------------------------------------
    # 3. Cảnh báo phát hiện mô hình thu hẹp biên độ (pattern_detector)
    # --------------------------------------------------------------------
    def send_pattern_alert(
        self,
        symbol: str,
        confidence: float,
        accumulation_high: float,
        chat_id: Optional[str] = None,
    ) -> bool:
        text = (
            f"📐 [Mô hình thu hẹp biên độ] {symbol}: phát hiện mô hình tích "
            f"lũy với độ tin cậy {confidence * 100:.1f}%. Ngưỡng tham khảo "
            f"breakout: {accumulation_high:,.2f}."
        )
        return self._send(chat_id, text, category="pattern")

    # --------------------------------------------------------------------
    # 4. Cảnh báo danh mục mô phỏng vượt ngưỡng lãi/lỗ
    # --------------------------------------------------------------------
    def send_portfolio_threshold_alert(
        self,
        current_pnl_pct: float,
        threshold_pct: float,
        chat_id: Optional[str] = None,
    ) -> bool:
        direction = "LÃI" if current_pnl_pct >= 0 else "LỖ"
        text = (
            f"💼 [Danh mục mô phỏng] Đã {direction} {abs(current_pnl_pct):.2f}%, "
            f"vượt ngưỡng cảnh báo {threshold_pct:.2f}%."
        )
        return self._send(chat_id, text, category="portfolio_threshold")

    # --------------------------------------------------------------------
    # 5. Cảnh báo dữ liệu gián đoạn (data_collector)
    # --------------------------------------------------------------------
    def send_stale_data_alert(
        self,
        data_key: str,
        minutes_since_update: float,
        chat_id: Optional[str] = None,
    ) -> bool:
        text = (
            f"⚠️ [Dữ liệu gián đoạn] '{data_key}' không được cập nhật trong "
            f"{minutes_since_update:.0f} phút qua."
        )
        return self._send(chat_id, text, category="stale_data")

    # --------------------------------------------------------------------
    # Xử lý lệnh điều khiển ĐI VÀO — BẮT BUỘC kiểm tra whitelist
    # --------------------------------------------------------------------
    def handle_incoming_command(self, chat_id: str, command: str) -> dict:
        """Xử lý một lệnh điều khiển gửi ĐẾN hệ thống (ví dụ /pause,
        /status) từ Telegram. BẮT BUỘC kiểm tra `chat_id` có nằm trong
        whitelist hay không TRƯỚC KHI thực thi bất kỳ điều gì — không tin
        bất kỳ người gửi lạ nào.

        Trả về dict: {"authorized": bool, "command": str, "response": str}.
        Nếu KHÔNG được whitelist, `authorized=False` và lệnh KHÔNG được
        thực thi dưới bất kỳ hình thức nào.
        """
        if chat_id not in self.whitelist_chat_ids:
            return {
                "authorized": False,
                "command": command,
                "response": (
                    "Từ chối: chat_id này không nằm trong whitelist được phép "
                    "gửi lệnh điều khiển."
                ),
            }

        # Whitelist hợp lệ -> thực thi lệnh (logic thực thi cụ thể cho từng
        # lệnh /pause, /status... sẽ được bổ sung khi tích hợp với main.py
        # và các module khác; ở đây chỉ xác nhận đã được phép).
        return {
            "authorized": True,
            "command": command,
            "response": f"Đã nhận lệnh '{command}' từ chat_id được whitelist.",
        }

    # --------------------------------------------------------------------
    # Tiện ích
    # --------------------------------------------------------------------
    def get_sent_log(self, category: Optional[str] = None) -> list[dict]:
        """Trả về lịch sử thông báo đã gửi, lọc theo `category` nếu có."""
        if category is None:
            return list(self.sent_log)
        return [entry for entry in self.sent_log if entry["category"] == category]
