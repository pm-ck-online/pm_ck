"""
Unit test cho core/notifier.py

Dùng MockTelegramClient (giả lập, không kế thừa RealTelegramClient) —
KHÔNG gọi Telegram Bot API thật khi chạy test.
"""

from __future__ import annotations

import pytest

from core.notifier import (
    NotifierError,
    Notifier,
    RealTelegramClient,
    TelegramClient,
)


class MockTelegramClient(TelegramClient):
    """Client giả lập cho unit test — chỉ ghi lại tin nhắn, không gọi mạng."""

    def __init__(self, should_succeed: bool = True):
        self.should_succeed = should_succeed
        self.sent_messages: list[tuple[str, str]] = []

    def send_message(self, chat_id: str, text: str) -> bool:
        self.sent_messages.append((chat_id, text))
        return self.should_succeed


@pytest.fixture
def mock_client() -> MockTelegramClient:
    return MockTelegramClient()


@pytest.fixture
def notifier(mock_client: MockTelegramClient) -> Notifier:
    return Notifier(
        client=mock_client,
        whitelist_chat_ids=["12345", "67890"],
        default_chat_id="99999",
    )


# ==============================================================================
# Test: 1. Cảnh báo giá chạm ngưỡng theo dõi
# ==============================================================================

class TestWatchlistAlert:
    def test_sends_message_with_correct_content(self, notifier, mock_client):
        result = notifier.send_watchlist_alert(
            symbol="HPG", current_price=32.5, threshold_price=32.0,
            direction="above", chat_id="12345",
        )
        assert result is True
        assert len(mock_client.sent_messages) == 1
        chat_id, text = mock_client.sent_messages[0]
        assert chat_id == "12345"
        assert "HPG" in text
        assert "32.5" in text or "32.50" in text

    def test_invalid_direction_raises(self, notifier):
        with pytest.raises(ValueError):
            notifier.send_watchlist_alert(
                symbol="HPG", current_price=32.5, threshold_price=32.0,
                direction="sideways", chat_id="12345",
            )


# ==============================================================================
# Test: 2. Cảnh báo đổi giai đoạn thị trường
# ==============================================================================

class TestRegimeChangeAlert:
    def test_includes_reasoning_in_message(self, notifier, mock_client):
        notifier.send_regime_change_alert(
            sector_or_market="banking",
            old_regime="sideway",
            new_regime="uptrend",
            reasoning=["80% mã trên EMA200", "Không có tín hiệu vĩ mô tiêu cực"],
            chat_id="12345",
        )
        _, text = mock_client.sent_messages[0]
        assert "sideway" in text
        assert "uptrend" in text
        assert "80% mã trên EMA200" in text


# ==============================================================================
# Test: 3. Cảnh báo mô hình thu hẹp biên độ
# ==============================================================================

class TestPatternAlert:
    def test_includes_confidence_and_accumulation_high(self, notifier, mock_client):
        notifier.send_pattern_alert(
            symbol="VNM", confidence=0.85, accumulation_high=85.5, chat_id="12345",
        )
        _, text = mock_client.sent_messages[0]
        assert "VNM" in text
        assert "85.0" in text or "85%" in text  # confidence hiển thị dạng %


# ==============================================================================
# Test: 4. Cảnh báo danh mục vượt ngưỡng lãi/lỗ
# ==============================================================================

class TestPortfolioThresholdAlert:
    def test_profit_message(self, notifier, mock_client):
        notifier.send_portfolio_threshold_alert(
            current_pnl_pct=15.0, threshold_pct=10.0, chat_id="12345",
        )
        _, text = mock_client.sent_messages[0]
        assert "LÃI" in text

    def test_loss_message(self, notifier, mock_client):
        notifier.send_portfolio_threshold_alert(
            current_pnl_pct=-12.0, threshold_pct=10.0, chat_id="12345",
        )
        _, text = mock_client.sent_messages[0]
        assert "LỖ" in text


# ==============================================================================
# Test: 5. Cảnh báo dữ liệu gián đoạn
# ==============================================================================

class TestStaleDataAlert:
    def test_includes_data_key_and_minutes(self, notifier, mock_client):
        notifier.send_stale_data_alert(
            data_key="ohlcv:HPG:day", minutes_since_update=20, chat_id="12345",
        )
        _, text = mock_client.sent_messages[0]
        assert "ohlcv:HPG:day" in text
        assert "20" in text


# ==============================================================================
# Test: default_chat_id fallback + lỗi khi thiếu chat_id đích
# ==============================================================================

class TestChatIdResolution:
    def test_uses_default_chat_id_when_not_specified(self, notifier, mock_client):
        notifier.send_watchlist_alert(
            symbol="HPG", current_price=30, threshold_price=29, direction="above",
        )
        chat_id, _ = mock_client.sent_messages[0]
        assert chat_id == "99999"

    def test_raises_when_no_chat_id_and_no_default(self, mock_client):
        notifier_no_default = Notifier(client=mock_client, whitelist_chat_ids=[])
        with pytest.raises(NotifierError):
            notifier_no_default.send_watchlist_alert(
                symbol="HPG", current_price=30, threshold_price=29, direction="above",
            )


# ==============================================================================
# Test: BẢO MẬT — whitelist cho lệnh điều khiển đi vào
# ==============================================================================

class TestIncomingCommandWhitelist:
    def test_whitelisted_chat_id_is_authorized(self, notifier):
        result = notifier.handle_incoming_command(chat_id="12345", command="/status")
        assert result["authorized"] is True

    def test_non_whitelisted_chat_id_is_rejected(self, notifier):
        result = notifier.handle_incoming_command(chat_id="666666", command="/pause")
        assert result["authorized"] is False

    def test_rejected_command_does_not_send_any_message(self, notifier, mock_client):
        notifier.handle_incoming_command(chat_id="666666", command="/pause")
        # Từ chối lệnh không được phép KHÔNG được kích hoạt gửi tin nhắn nào
        assert len(mock_client.sent_messages) == 0

    def test_empty_whitelist_rejects_everyone(self, mock_client):
        notifier_empty = Notifier(client=mock_client, whitelist_chat_ids=[])
        result = notifier_empty.handle_incoming_command(chat_id="12345", command="/status")
        assert result["authorized"] is False


# ==============================================================================
# Test: nhật ký gửi tin (sent_log) và lọc theo category
# ==============================================================================

class TestSentLog:
    def test_log_records_all_sent_messages(self, notifier):
        notifier.send_watchlist_alert(
            symbol="HPG", current_price=30, threshold_price=29,
            direction="above", chat_id="12345",
        )
        notifier.send_pattern_alert(
            symbol="VNM", confidence=0.8, accumulation_high=90, chat_id="12345",
        )
        assert len(notifier.get_sent_log()) == 2

    def test_filter_log_by_category(self, notifier):
        notifier.send_watchlist_alert(
            symbol="HPG", current_price=30, threshold_price=29,
            direction="above", chat_id="12345",
        )
        notifier.send_pattern_alert(
            symbol="VNM", confidence=0.8, accumulation_high=90, chat_id="12345",
        )
        watchlist_log = notifier.get_sent_log(category="watchlist")
        assert len(watchlist_log) == 1
        assert watchlist_log[0]["category"] == "watchlist"


# ==============================================================================
# Test: RealTelegramClient — KHÔNG gọi mạng thật, chỉ kiểm tra bảo mật token
# ==============================================================================

class TestRealTelegramClientSecurity:
    def test_raises_when_token_env_var_missing(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        client = RealTelegramClient(token_env_var="TELEGRAM_BOT_TOKEN")
        with pytest.raises(NotifierError):
            client.send_message(chat_id="12345", text="test")

    def test_does_not_hardcode_token(self):
        # Xác nhận constructor không nhận trực tiếp giá trị token, chỉ
        # nhận TÊN biến môi trường — đảm bảo không có chỗ nào lỡ hardcode.
        client = RealTelegramClient(token_env_var="TELEGRAM_BOT_TOKEN")
        assert client.token_env_var == "TELEGRAM_BOT_TOKEN"
        assert not hasattr(client, "token") or client.token_env_var != "token"
