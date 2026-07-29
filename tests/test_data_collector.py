"""
Unit test cho core/data_collector.py

Toàn bộ test dùng MockDataSource (dữ liệu giả lập) — KHÔNG gọi API thật.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta

import pandas as pd
import pytest

from core.data_collector import (
    DataCollector,
    DataSourceError,
    FundamentalData,
    MacroDataPoint,
    MockDataSource,
    NewsItem,
    RetryExhaustedError,
    VnstockDataSource,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_source() -> MockDataSource:
    return MockDataSource()


@pytest.fixture
def collector(mock_source: MockDataSource) -> DataCollector:
    return DataCollector(mock_source)


# ==============================================================================
# Test: get_ohlcv
# ==============================================================================

class TestGetOHLCV:
    def test_returns_dataframe_with_expected_columns(self, collector: DataCollector):
        df = collector.get_ohlcv("HPG", timeframe="day")
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"date", "open", "high", "low", "close", "volume"}
        assert expected_cols.issubset(set(df.columns))

    def test_returns_non_empty_data(self, collector: DataCollector):
        df = collector.get_ohlcv("VNM", timeframe="day")
        assert len(df) > 0

    def test_weekly_timeframe_returns_fewer_rows_than_daily(self, collector: DataCollector):
        daily = collector.get_ohlcv("FPT", timeframe="day")
        weekly = collector.get_ohlcv("FPT", timeframe="week")
        assert len(weekly) < len(daily)

    def test_deterministic_for_same_symbol(self, collector: DataCollector):
        df1 = collector.get_ohlcv("MWG", timeframe="day", use_cache=False)
        df2 = collector.get_ohlcv("MWG", timeframe="day", use_cache=False)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_symbols_produce_different_data(self, collector: DataCollector):
        df_a = collector.get_ohlcv("AAA", timeframe="day")
        df_b = collector.get_ohlcv("ZZZ", timeframe="day")
        assert not df_a["close"].equals(df_b["close"])


# ==============================================================================
# Test: Cache
# ==============================================================================

class TestCache:
    def test_second_call_uses_cache_not_source(self, mock_source: MockDataSource):
        collector = DataCollector(mock_source, config={"cache": {"enabled": True}})
        collector.get_ohlcv("HPG", timeframe="day")
        call_count_after_first = mock_source._call_counts.get("ohlcv:HPG:day", 0)

        collector.get_ohlcv("HPG", timeframe="day")
        call_count_after_second = mock_source._call_counts.get("ohlcv:HPG:day", 0)

        # Lần gọi thứ 2 phải lấy từ cache -> số lần gọi source KHÔNG tăng thêm
        assert call_count_after_first == call_count_after_second

    def test_use_cache_false_bypasses_cache(self, mock_source: MockDataSource):
        collector = DataCollector(mock_source, config={"cache": {"enabled": True}})
        collector.get_ohlcv("HPG", timeframe="day", use_cache=True)
        collector.get_ohlcv("HPG", timeframe="day", use_cache=False)

        call_count = mock_source._call_counts.get("ohlcv:HPG:day", 0)
        assert call_count == 2

    def test_cache_disabled_always_calls_source(self, mock_source: MockDataSource):
        collector = DataCollector(mock_source, config={"cache": {"enabled": False}})
        collector.get_ohlcv("HPG", timeframe="day")
        collector.get_ohlcv("HPG", timeframe="day")

        call_count = mock_source._call_counts.get("ohlcv:HPG:day", 0)
        assert call_count == 2


# ==============================================================================
# Test: Retry
# ==============================================================================

class TestRetry:
    def test_succeeds_after_transient_failures_within_max_attempts(self):
        source = MockDataSource(fail_times=2)  # 2 lần đầu lỗi, lần 3 thành công
        collector = DataCollector(
            source, config={"retry": {"max_attempts": 3, "backoff_seconds": 0}}
        )
        df = collector.get_ohlcv("HPG", timeframe="day")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_raises_after_exhausting_retries(self):
        source = MockDataSource(fail_times=5)  # luôn lỗi, vượt quá max_attempts
        collector = DataCollector(
            source, config={"retry": {"max_attempts": 3, "backoff_seconds": 0}}
        )
        with pytest.raises(RetryExhaustedError):
            collector.get_ohlcv("HPG", timeframe="day")


# ==============================================================================
# Test: Giá thời gian thực
# ==============================================================================

class TestRealtimePrice:
    def test_returns_expected_keys(self, collector: DataCollector):
        result = collector.get_realtime_price("HPG")
        assert set(result.keys()) == {"symbol", "price", "volume", "timestamp"}
        assert result["symbol"] == "HPG"
        assert isinstance(result["price"], float)


# ==============================================================================
# Test: Giờ giao dịch
# ==============================================================================

class TestTradingHours:
    def test_within_morning_session(self, collector: DataCollector):
        # Thứ Hai (weekday=0), 10:00 sáng
        dt = datetime(2026, 7, 20, 10, 0)
        assert dt.weekday() == 0
        assert collector.is_trading_hours(dt) is True

    def test_within_afternoon_session(self, collector: DataCollector):
        dt = datetime(2026, 7, 20, 14, 0)
        assert collector.is_trading_hours(dt) is True

    def test_outside_trading_hours_lunch_break(self, collector: DataCollector):
        dt = datetime(2026, 7, 20, 12, 0)
        assert collector.is_trading_hours(dt) is False

    def test_outside_trading_hours_evening(self, collector: DataCollector):
        dt = datetime(2026, 7, 20, 20, 0)
        assert collector.is_trading_hours(dt) is False

    def test_weekend_is_not_trading_hours(self, collector: DataCollector):
        # Chủ nhật
        dt = datetime(2026, 7, 26, 10, 0)
        assert dt.weekday() == 6
        assert collector.is_trading_hours(dt) is False


# ==============================================================================
# Test: Dữ liệu cơ bản doanh nghiệp
# ==============================================================================

class TestFundamentals:
    def test_returns_fundamental_data_object(self, collector: DataCollector):
        result = collector.get_fundamentals("HPG")
        assert isinstance(result, FundamentalData)
        assert result.symbol == "HPG"
        assert result.eps is not None
        assert result.pe is not None


# ==============================================================================
# Test: Tin tức
# ==============================================================================

class TestNews:
    def test_returns_list_of_news_items(self, collector: DataCollector):
        news = collector.get_news("HPG")
        assert isinstance(news, list)
        assert len(news) > 0
        assert all(isinstance(n, NewsItem) for n in news)

    def test_news_without_symbol_returns_general_news(self, collector: DataCollector):
        news = collector.get_news()
        assert isinstance(news, list)
        assert len(news) > 0


# ==============================================================================
# Test: Dữ liệu vĩ mô (nhóm ưu tiên cao nhất)
# ==============================================================================

class TestMacroData:
    def test_returns_list_of_macro_points(self, collector: DataCollector):
        macro = collector.get_macro_data()
        assert isinstance(macro, list)
        assert len(macro) > 0
        assert all(isinstance(m, MacroDataPoint) for m in macro)

    def test_macro_points_have_categories(self, collector: DataCollector):
        macro = collector.get_macro_data()
        categories = {m.category for m in macro}
        # Đảm bảo có đủ các nhóm dữ liệu vĩ mô ưu tiên theo yêu cầu dự án
        assert "fx_intervention" in categories
        assert "omo" in categories
        assert "interest_rate" in categories
        assert "sector_policy" in categories

    def test_get_macro_data_by_sector_filters_correctly(self, collector: DataCollector):
        real_estate_macro = collector.get_macro_data_by_sector("real_estate")
        assert len(real_estate_macro) > 0
        assert all("real_estate" in m.affected_sectors for m in real_estate_macro)

    def test_get_macro_data_by_sector_empty_for_unrelated_sector(
        self, collector: DataCollector
    ):
        result = collector.get_macro_data_by_sector("technology_unrelated_sector")
        assert result == []


# ==============================================================================
# Test: Cảnh báo nguồn dữ liệu trả phí
# ==============================================================================

class TestPaidSourceWarning:
    class _PaidMockSource(MockDataSource):
        name = "paid_mock"
        is_paid_source = True

    def test_warns_when_paid_source_and_not_delayed(self, caplog):
        source = self._PaidMockSource()
        with caplog.at_level("WARNING", logger="pm_ck.data_collector"):
            DataCollector(source, config={"delayed_mode": False})
        assert any("TÍNH PHÍ" in record.message for record in caplog.records)

    def test_no_warning_when_paid_source_but_delayed_mode_on(self, caplog):
        source = self._PaidMockSource()
        with caplog.at_level("WARNING", logger="pm_ck.data_collector"):
            DataCollector(source, config={"delayed_mode": True})
        assert not any("TÍNH PHÍ" in record.message for record in caplog.records)

    def test_no_warning_for_free_mock_source(self, caplog):
        with caplog.at_level("WARNING", logger="pm_ck.data_collector"):
            DataCollector(MockDataSource(), config={"delayed_mode": False})
        assert not any("TÍNH PHÍ" in record.message for record in caplog.records)


# ==============================================================================
# Test: Dữ liệu gián đoạn (stale data)
# ==============================================================================

class TestStaleDataCheck:
    def test_not_stale_immediately_after_fetch(self, collector: DataCollector):
        collector.get_ohlcv("HPG", timeframe="day")
        assert collector.check_stale("ohlcv:HPG:day") is False

    def test_unknown_key_is_not_considered_stale(self, collector: DataCollector):
        assert collector.check_stale("khong_ton_tai") is False

    def test_stale_when_last_update_too_old(self, collector: DataCollector):
        collector.get_ohlcv("HPG", timeframe="day")
        # Giả lập lần cập nhật gần nhất đã lâu hơn ngưỡng cảnh báo
        collector._last_update_at["ohlcv:HPG:day"] = datetime.now() - timedelta(
            minutes=999
        )
        assert collector.check_stale("ohlcv:HPG:day") is True


# ==============================================================================
# Test: VnstockDataSource (adapter dữ liệu thật)
# ==============================================================================
# Dùng module `vnstock` GIẢ LẬP (tiêm vào sys.modules) — KHÔNG cài đặt
# vnstock thật, KHÔNG gọi mạng thật. Cấu trúc dữ liệu giả lập mô phỏng
# đúng theo kết quả thực tế đã xác minh (xem docstring VnstockDataSource).

class TestVnstockDataSource:
    @pytest.fixture
    def fake_vnstock_module(self, monkeypatch):
        """Tiêm một module `vnstock` giả vào sys.modules, với class `Market`
        giả lập trả về dữ liệu có cấu trúc giống thật (đã xác minh thủ công).
        """
        fake_module = types.ModuleType("vnstock")

        class FakeEquity:
            def ohlcv(self, interval, count):
                return pd.DataFrame({
                    "time": pd.bdate_range("2025-01-01", periods=5),
                    "open": [25.3, 25.4, 25.1, 25.3, 25.2],
                    "high": [25.9, 25.5, 25.3, 25.7, 25.7],
                    "low": [25.1, 24.8, 24.9, 24.8, 25.1],
                    "close": [25.4, 25.0, 25.1, 24.9, 25.3],
                    "volume": [84195900, 79765000, 76049900, 107927600, 69044400],
                })

        class FakeIndex:
            def ohlcv(self, interval, count):
                return pd.DataFrame({
                    "time": pd.bdate_range("2025-01-01", periods=5),
                    "open": [1200.0, 1205.0, 1198.0, 1210.0, 1215.0],
                    "high": [1210.0, 1212.0, 1205.0, 1218.0, 1220.0],
                    "low": [1195.0, 1198.0, 1190.0, 1205.0, 1208.0],
                    "close": [1205.0, 1198.0, 1210.0, 1215.0, 1218.0],
                    "volume": [500_000_000, 480_000_000, 520_000_000, 510_000_000, 530_000_000],
                })

        class FakeMarket:
            def equity(self, symbol):
                return FakeEquity()

            def index(self, symbol):
                return FakeIndex()

            def quote(self, symbol):
                return pd.DataFrame([{
                    "symbol": symbol,
                    "time": 1784622792999,
                    "exchange": "HOSE",
                    "close_price": 20800,
                    "volume_accumulated": 27198700,
                    # ... các cột khác không cần thiết cho test này
                }])

        fake_module.Market = FakeMarket
        monkeypatch.setitem(sys.modules, "vnstock", fake_module)
        return fake_module

    def test_fetch_ohlcv_renames_time_to_date(self, fake_vnstock_module):
        source = VnstockDataSource()
        df = source.fetch_ohlcv("HPG", timeframe="day")

        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert len(df) == 5

    def test_fetch_ohlcv_raises_on_missing_columns(self, monkeypatch):
        fake_module = types.ModuleType("vnstock")

        class BadEquity:
            def ohlcv(self, interval, count):
                return pd.DataFrame({"time": [1], "open": [1]})  # thiếu nhiều cột

        class BadMarket:
            def equity(self, symbol):
                return BadEquity()

        fake_module.Market = BadMarket
        monkeypatch.setitem(sys.modules, "vnstock", fake_module)

        source = VnstockDataSource()
        with pytest.raises(DataSourceError):
            source.fetch_ohlcv("HPG")

    def test_fetch_realtime_price_parses_correctly(self, fake_vnstock_module):
        source = VnstockDataSource()
        result = source.fetch_realtime_price("HPG")

        assert result["symbol"] == "HPG"
        assert result["price"] == pytest.approx(20800)
        assert result["volume"] == pytest.approx(27198700)
        assert isinstance(result["timestamp"], datetime)

    def test_fetch_realtime_price_raises_on_missing_columns(self, monkeypatch):
        fake_module = types.ModuleType("vnstock")

        class BadMarket:
            def quote(self, symbol):
                return pd.DataFrame([{"symbol": symbol}])  # thiếu cột giá

        fake_module.Market = BadMarket
        monkeypatch.setitem(sys.modules, "vnstock", fake_module)

        source = VnstockDataSource()
        with pytest.raises(DataSourceError):
            source.fetch_realtime_price("HPG")

    def test_raises_clear_error_when_vnstock_not_installed(self, monkeypatch):
        # Đặt sys.modules["vnstock"] = None là cách buộc Python raise
        # ImportError ngay lập tức khi import, BẤT KỂ máy đang chạy test có
        # thực sự cài đặt gói `vnstock` hay không (khác với việc chỉ xóa
        # khỏi sys.modules — trường hợp đó nếu gói thật đã cài trên đĩa,
        # Python vẫn import lại được bình thường, không mô phỏng đúng tình
        # huống "chưa cài đặt").
        monkeypatch.setitem(sys.modules, "vnstock", None)
        source = VnstockDataSource()
        with pytest.raises(DataSourceError, match="vnstock"):
            source.fetch_ohlcv("HPG")

    def test_fetch_macro_data_returns_empty_list_with_warning(self, fake_vnstock_module, caplog):
        source = VnstockDataSource()
        with caplog.at_level("WARNING", logger="pm_ck.data_collector"):
            result = source.fetch_macro_data()
        assert result == []
        assert any("vĩ mô" in r.message for r in caplog.records)

    def test_fetch_fundamentals_raises_not_implemented(self, fake_vnstock_module):
        source = VnstockDataSource()
        with pytest.raises(NotImplementedError):
            source.fetch_fundamentals("HPG")

    def test_fetch_news_raises_not_implemented(self, fake_vnstock_module):
        source = VnstockDataSource()
        with pytest.raises(NotImplementedError):
            source.fetch_news("HPG")

    def test_fetch_symbol_sector_map_returns_correct_mapping(self, monkeypatch):
        fake_module = types.ModuleType("vnstock")

        class FakeListing:
            def __init__(self, source=None):
                self.source = source

            def symbols_by_industries(self):
                return pd.DataFrame({
                    "symbol": ["HPG", "VNM", "FPT"],
                    "industry_code": ["1010", "2020", "3030"],
                    "industry_name": ["Tài nguyên Cơ bản", "Thực phẩm & Đồ uống", "Công nghệ Thông tin"],
                })

        fake_module.Listing = FakeListing
        monkeypatch.setitem(sys.modules, "vnstock", fake_module)

        source = VnstockDataSource()
        result = source.fetch_symbol_sector_map()

        assert result == {
            "HPG": "Tài nguyên Cơ bản",
            "VNM": "Thực phẩm & Đồ uống",
            "FPT": "Công nghệ Thông tin",
        }

    def test_fetch_symbol_sector_map_raises_on_missing_columns(self, monkeypatch):
        fake_module = types.ModuleType("vnstock")

        class BadListing:
            def __init__(self, source=None):
                self.source = source

            def symbols_by_industries(self):
                return pd.DataFrame({"symbol": ["HPG"]})  # thiếu industry_name

        fake_module.Listing = BadListing
        monkeypatch.setitem(sys.modules, "vnstock", fake_module)

        source = VnstockDataSource()
        with pytest.raises(DataSourceError):
            source.fetch_symbol_sector_map()

    def test_fetch_index_ohlcv_renames_time_to_date(self, fake_vnstock_module):
        source = VnstockDataSource()
        df = source.fetch_index_ohlcv("VNINDEX", timeframe="day")

        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert len(df) == 5

    def test_fetch_index_ohlcv_raises_on_missing_columns(self, monkeypatch):
        fake_module = types.ModuleType("vnstock")

        class BadIndex:
            def ohlcv(self, interval, count):
                return pd.DataFrame({"time": [1], "open": [1]})  # thiếu nhiều cột

        class BadMarket:
            def index(self, symbol):
                return BadIndex()

        fake_module.Market = BadMarket
        monkeypatch.setitem(sys.modules, "vnstock", fake_module)

        source = VnstockDataSource()
        with pytest.raises(DataSourceError):
            source.fetch_index_ohlcv("VNINDEX")

    def test_get_index_ohlcv_uses_index_specific_endpoint(self, fake_vnstock_module):
        source = VnstockDataSource()
        collector = DataCollector(source)
        df = collector.get_index_ohlcv("VN30", timeframe="day")

        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert df["close"].iloc[0] == pytest.approx(1205.0)  # khớp dữ liệu FakeIndex, không phải FakeEquity

    def test_default_fetch_index_ohlcv_falls_back_to_fetch_ohlcv_for_mock(self):
        # MockDataSource không override fetch_index_ohlcv -> dùng lại fetch_ohlcv() mặc định
        source = MockDataSource()
        df_index = source.fetch_index_ohlcv("VNINDEX")
        df_equity = source.fetch_ohlcv("VNINDEX")
        pd.testing.assert_frame_equal(df_index, df_equity)
