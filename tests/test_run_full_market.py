"""
Unit test cho run_full_market.py

Dùng module `vnstock` giả lập (không gọi mạng thật) và giả lập
`time.sleep` (không chờ thật) để test chạy nhanh.

Dùng file SQLite THẬT (qua `tmp_path`) thay vì ":memory:" — vì
`run_full_market()` tự đóng kết nối storage khi chạy xong, nên cần mở
lại bằng một `Storage` mới trỏ tới CÙNG file để kiểm tra kết quả sau đó.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from core.storage import Storage
from run_full_market import (
    compute_capital_allocations_for_all_symbols,
    load_checkpoint,
    run_full_market,
    save_checkpoint,
)


@pytest.fixture
def fake_vnstock_module(monkeypatch):
    """Module vnstock giả lập: 5 mã, mỗi mã trả về OHLCV hợp lệ."""
    fake_module = types.ModuleType("vnstock")

    class FakeEquity:
        def ohlcv(self, interval, count):
            return pd.DataFrame({
                "time": pd.bdate_range("2024-01-01", periods=260),
                "open": [10.0 + (i % 5) for i in range(260)],
                "high": [10.5 + (i % 5) for i in range(260)],
                "low": [9.5 + (i % 5) for i in range(260)],
                "close": [10.0 + (i % 5) for i in range(260)],
                "volume": [100000 + i for i in range(260)],
            })

    class FakeMarket:
        def equity(self, symbol):
            return FakeEquity()

        def quote(self, symbol):
            return pd.DataFrame([{
                "symbol": symbol, "time": 1784622792999,
                "close_price": 10.0, "volume_accumulated": 100000,
            }])

    class FakeListing:
        def __init__(self, source=None):
            pass

        def symbols_by_industries(self):
            return pd.DataFrame({
                "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                "industry_code": ["1", "2", "3", "4", "5"],
                "industry_name": ["Ngành A", "Ngành B", "Ngành C", "Ngành D", "Ngành E"],
            })

    fake_module.Market = FakeMarket
    fake_module.Listing = FakeListing
    monkeypatch.setitem(sys.modules, "vnstock", fake_module)
    return fake_module


@pytest.fixture
def no_sleep(monkeypatch):
    """Giả lập time.sleep để test chạy nhanh, không chờ thật."""
    monkeypatch.setattr("run_full_market.time.sleep", lambda seconds: None)


@pytest.fixture
def config_with_db(tmp_path):
    """Cấu hình trỏ tới 1 file SQLite thật trong thư mục tạm của pytest.

    Tắt retry nội bộ của DataCollector (max_attempts=1, backoff=0) để các
    test về rate limit dưới đây kiểm tra ĐÚNG cơ chế retry mới ở tầng
    `run_full_market` (không bị lẫn với retry có sẵn của DataCollector,
    vốn có backoff chờ thật bằng giây).
    """
    db_path = str(tmp_path / "test_pm_ck.db")
    return {
        "storage": {"path": db_path},
        "data_source": {
            "delayed_mode": True,
            "cache": {"enabled": False},
            "retry": {"max_attempts": 1, "backoff_seconds": 0},
        },
        "indicators": {},
        "pattern_detector": {"scan_months_min": 10, "scan_months_max": 30, "n_segments": 4},
    }, db_path


# ==============================================================================
# Test: checkpoint (dùng :memory: trực tiếp — không qua run_full_market nên an toàn)
# ==============================================================================

class TestCheckpoint:
    def test_load_checkpoint_empty_when_no_data(self):
        storage = Storage(db_path=":memory:")
        result = load_checkpoint(storage)
        assert result == set()
        storage.close()

    def test_save_and_load_checkpoint_roundtrip(self):
        storage = Storage(db_path=":memory:")
        save_checkpoint(storage, {"AAA", "BBB"})
        result = load_checkpoint(storage)
        assert result == {"AAA", "BBB"}
        storage.close()


# ==============================================================================
# Test: run_full_market — chạy toàn bộ, giới hạn, resume, lỗi 1 mã
# ==============================================================================

class TestRunFullMarket:
    def test_processes_all_symbols_and_saves_data(
        self, fake_vnstock_module, no_sleep, config_with_db
    ):
        config, db_path = config_with_db
        run_full_market(config, delay_seconds=0)

        storage = Storage(db_path=db_path)
        completed = load_checkpoint(storage)
        assert completed == {"AAA", "BBB", "CCC", "DDD", "EEE"}

        record = storage.get_latest("indicator_snapshot", "AAA")
        assert record is not None
        storage.close()

    def test_limit_restricts_number_of_symbols_processed(
        self, fake_vnstock_module, no_sleep, config_with_db
    ):
        config, db_path = config_with_db
        run_full_market(config, delay_seconds=0, limit=2)

        storage = Storage(db_path=db_path)
        completed = load_checkpoint(storage)
        assert len(completed) == 2
        storage.close()

    def test_resume_skips_already_completed_symbols(
        self, fake_vnstock_module, no_sleep, config_with_db
    ):
        config, db_path = config_with_db

        # Giả lập đã chạy trước đó, hoàn thành 3/5 mã
        setup_storage = Storage(db_path=db_path)
        save_checkpoint(setup_storage, {"AAA", "BBB", "CCC"})
        setup_storage.close()

        run_full_market(config, delay_seconds=0)

        storage = Storage(db_path=db_path)
        completed = load_checkpoint(storage)
        # Sau khi resume: đủ cả 5 mã (3 cũ + 2 mới xử lý thêm)
        assert completed == {"AAA", "BBB", "CCC", "DDD", "EEE"}
        storage.close()

    def test_reset_ignores_previous_checkpoint(
        self, fake_vnstock_module, no_sleep, config_with_db
    ):
        config, db_path = config_with_db

        setup_storage = Storage(db_path=db_path)
        save_checkpoint(setup_storage, {"AAA", "BBB", "CCC", "DDD", "EEE"})  # coi như đã xong hết
        setup_storage.close()

        # reset=True + limit=1 -> phải xử lý lại ít nhất 1 mã dù checkpoint đã đầy đủ
        run_full_market(config, delay_seconds=0, limit=1, reset_checkpoint=True)

        storage = Storage(db_path=db_path)
        completed = load_checkpoint(storage)
        # Với reset, checkpoint mới chỉ có tối đa 1 mã (do limit=1)
        assert len(completed) == 1
        storage.close()

    def test_one_symbol_error_does_not_stop_batch(
        self, monkeypatch, no_sleep, config_with_db
    ):
        config, db_path = config_with_db

        fake_module = types.ModuleType("vnstock")

        class FlakyEquity:
            def __init__(self, symbol):
                self.symbol = symbol

            def ohlcv(self, interval, count):
                if self.symbol == "BBB":
                    raise RuntimeError("Lỗi giả lập cho mã BBB")
                return pd.DataFrame({
                    "time": pd.bdate_range("2024-01-01", periods=260),
                    "open": [10.0] * 260, "high": [10.5] * 260,
                    "low": [9.5] * 260, "close": [10.0] * 260,
                    "volume": [100000] * 260,
                })

        class FakeMarket:
            def equity(self, symbol):
                return FlakyEquity(symbol)

            def quote(self, symbol):
                return pd.DataFrame([{
                    "symbol": symbol, "time": 1784622792999,
                    "close_price": 10.0, "volume_accumulated": 100000,
                }])

        class FakeListing:
            def __init__(self, source=None):
                pass

            def symbols_by_industries(self):
                return pd.DataFrame({
                    "symbol": ["AAA", "BBB", "CCC"],
                    "industry_code": ["1", "2", "3"],
                    "industry_name": ["Ngành A", "Ngành B", "Ngành C"],
                })

        fake_module.Market = FakeMarket
        fake_module.Listing = FakeListing
        monkeypatch.setitem(sys.modules, "vnstock", fake_module)

        run_full_market(config, delay_seconds=0)

        storage = Storage(db_path=db_path)
        completed = load_checkpoint(storage)
        # Cả 3 mã đều nằm trong checkpoint (kể cả BBB bị lỗi -> coi như đã xử lý, bỏ qua)
        assert completed == {"AAA", "BBB", "CCC"}
        # Nhưng chỉ AAA và CCC thực sự có dữ liệu chỉ báo được lưu
        assert storage.get_latest("indicator_snapshot", "AAA") is not None
        assert storage.get_latest("indicator_snapshot", "BBB") is None
        assert storage.get_latest("indicator_snapshot", "CCC") is not None
        storage.close()


# ==============================================================================
# Test: aggregate_market_regime_by_sector (tổng hợp giai đoạn thị trường theo ngành)
# ==============================================================================

class TestAggregateMarketRegimeBySector:
    def test_groups_symbols_by_real_sector_and_computes_breadth(
        self, fake_vnstock_module, no_sleep, config_with_db
    ):
        config, db_path = config_with_db
        # FakeListing trong fake_vnstock_module gán mỗi mã 1 ngành riêng
        # (Ngành A..E, 1 mã/ngành) -> chạy full market rồi kiểm tra bước
        # tổng hợp tạo đúng bản ghi market_regime_quant cho từng ngành.
        run_full_market(config, delay_seconds=0)

        storage = Storage(db_path=db_path)
        sectors = storage.query_all_keys("market_regime_quant")
        assert set(sectors) == {"Ngành A", "Ngành B", "Ngành C", "Ngành D", "Ngành E"}

        record = storage.get_latest("market_regime_quant", "Ngành A")
        assert record["data"]["breadth_pct"] is not None
        assert record["data"]["trang_thai"] in {"UPTREND", "DOWNTREND", "SIDEWAY"}

        # Rà soát sự cố thực tế 28/07/2026: mục "Giai đoạn thị trường
        # (định tính)" trên dashboard đọc category "market_regime" —
        # trước đây run_full_market.py KHÔNG BAO GIỜ ghi vào category
        # này (chỉ main.py ghi), khiến mục đó luôn trống khi dùng
        # run_full_market.py. Phải ghi ĐỦ CẢ 2 category.
        qualitative_sectors = storage.query_all_keys("market_regime")
        assert set(qualitative_sectors) == {"Ngành A", "Ngành B", "Ngành C", "Ngành D", "Ngành E"}
        qualitative_record = storage.get_latest("market_regime", "Ngành A")
        assert qualitative_record["data"]["regime"] in {"uptrend", "downtrend", "sideway", None}
        storage.close()

    def test_aggregates_multiple_symbols_into_same_sector(
        self, monkeypatch, no_sleep, config_with_db
    ):
        config, db_path = config_with_db

        fake_module = types.ModuleType("vnstock")

        class FakeEquity:
            def __init__(self, symbol):
                self.symbol = symbol

            def ohlcv(self, interval, count):
                # AAA, BBB có giá trên EMA200 (uptrend); CCC dưới EMA200 (downtrend)
                base = 200.0 if self.symbol != "CCC" else 50.0
                return pd.DataFrame({
                    "time": pd.bdate_range("2024-01-01", periods=260),
                    "open": [base] * 260, "high": [base * 1.01] * 260,
                    "low": [base * 0.99] * 260, "close": [base] * 260,
                    "volume": [100000] * 260,
                })

        class FakeMarket:
            def equity(self, symbol):
                return FakeEquity(symbol)

            def quote(self, symbol):
                return pd.DataFrame([{
                    "symbol": symbol, "time": 1784622792999,
                    "close_price": 100.0, "volume_accumulated": 100000,
                }])

        class FakeListing:
            def __init__(self, source=None):
                pass

            def symbols_by_industries(self):
                # AAA và BBB CÙNG 1 ngành ("Bán lẻ"), CCC ngành khác
                return pd.DataFrame({
                    "symbol": ["AAA", "BBB", "CCC"],
                    "industry_code": ["1", "1", "2"],
                    "industry_name": ["Bán lẻ", "Bán lẻ", "Vật liệu xây dựng"],
                })

        fake_module.Market = FakeMarket
        fake_module.Listing = FakeListing
        monkeypatch.setitem(sys.modules, "vnstock", fake_module)

        run_full_market(config, delay_seconds=0)

        storage = Storage(db_path=db_path)
        record = storage.get_latest("market_regime_quant", "Bán lẻ")
        # Cả 2 mã AAA, BBB đều được gộp vào breadth của ngành "Bán lẻ"
        assert record["data"]["breadth_pct"] is not None
        storage.close()


# ==============================================================================
# Test: xử lý lỗi giới hạn API (rate limit) — bao gồm cả trường hợp
# thư viện raise SystemExit thay vì Exception thông thường (đã xảy ra
# trong thực tế khi kiểm thử với vnstock gói Khách/Guest).
# ==============================================================================

class TestRateLimitHandling:
    def _make_flaky_rate_limit_module(self, fail_times: int, raise_system_exit: bool = False):
        """Tạo module vnstock giả lập: mã 'AAA' bị lỗi rate limit
        `fail_times` lần đầu, sau đó thành công (nếu fail_times đủ nhỏ).
        """
        call_count = {"AAA": 0}

        class FlakyEquity:
            def ohlcv(self, interval, count):
                call_count["AAA"] += 1
                if call_count["AAA"] <= fail_times:
                    message = "Rate limit exceeded. Đã đạt giới hạn tối đa."
                    if raise_system_exit:
                        raise SystemExit(message)
                    raise RuntimeError(message)
                return pd.DataFrame({
                    "time": pd.bdate_range("2024-01-01", periods=260),
                    "open": [10.0] * 260, "high": [10.5] * 260,
                    "low": [9.5] * 260, "close": [10.0] * 260,
                    "volume": [100000] * 260,
                })

        class FakeMarket:
            def equity(self, symbol):
                return FlakyEquity()

            def quote(self, symbol):
                return pd.DataFrame([{
                    "symbol": symbol, "time": 1784622792999,
                    "close_price": 10.0, "volume_accumulated": 100000,
                }])

        class FakeListing:
            def __init__(self, source=None):
                pass

            def symbols_by_industries(self):
                return pd.DataFrame({
                    "symbol": ["AAA"],
                    "industry_code": ["1"],
                    "industry_name": ["Ngành A"],
                })

        fake_module = types.ModuleType("vnstock")
        fake_module.Market = FakeMarket
        fake_module.Listing = FakeListing
        return fake_module

    def test_retries_after_rate_limit_error_and_succeeds(
        self, monkeypatch, no_sleep, config_with_db
    ):
        config, db_path = config_with_db
        fake_module = self._make_flaky_rate_limit_module(fail_times=2)
        monkeypatch.setitem(sys.modules, "vnstock", fake_module)

        run_full_market(config, delay_seconds=0, max_rate_limit_retries=5,
                         rate_limit_cooldown_seconds=0)

        storage = Storage(db_path=db_path)
        completed = load_checkpoint(storage)
        assert completed == {"AAA"}
        assert storage.get_latest("indicator_snapshot", "AAA") is not None
        storage.close()

    def test_catches_system_exit_raised_by_rate_limit(
        self, monkeypatch, no_sleep, config_with_db
    ):
        """Mô phỏng ĐÚNG tình huống thực tế: vnstock raise SystemExit khi
        vượt giới hạn API — code PHẢI bắt được, không được để sập cả
        chương trình.
        """
        config, db_path = config_with_db
        fake_module = self._make_flaky_rate_limit_module(fail_times=1, raise_system_exit=True)
        monkeypatch.setitem(sys.modules, "vnstock", fake_module)

        # Không được raise ra ngoài — nếu code không bắt SystemExit đúng
        # cách, dòng dưới đây sẽ tự nhảy ra SystemExit và test FAIL.
        run_full_market(config, delay_seconds=0, max_rate_limit_retries=5,
                         rate_limit_cooldown_seconds=0)

        storage = Storage(db_path=db_path)
        completed = load_checkpoint(storage)
        assert completed == {"AAA"}
        storage.close()

    def test_gives_up_after_max_retries_without_marking_completed(
        self, monkeypatch, no_sleep, config_with_db
    ):
        config, db_path = config_with_db
        # Luôn luôn lỗi rate limit, không bao giờ thành công
        fake_module = self._make_flaky_rate_limit_module(fail_times=999)
        monkeypatch.setitem(sys.modules, "vnstock", fake_module)

        run_full_market(config, delay_seconds=0, max_rate_limit_retries=2,
                         rate_limit_cooldown_seconds=0)

        storage = Storage(db_path=db_path)
        completed = load_checkpoint(storage)
        # KHÔNG được đánh dấu hoàn thành -> lần chạy sau sẽ tự thử lại
        assert completed == set()
        assert storage.get_latest("indicator_snapshot", "AAA") is None
        storage.close()


class TestCustomWatchlistOverride:
    def test_uses_custom_symbols_from_config_instead_of_full_market(
        self, fake_vnstock_module, no_sleep, config_with_db
    ):
        """Khi config.yaml có watchlist.symbols -> PHẢI dùng đúng danh
        sách + ngành đó, KHÔNG gọi fetch_symbol_sector_map() (quét toàn
        bộ thị trường theo phân loại vnstock).
        """
        config, db_path = config_with_db
        config["watchlist"] = {"symbols": {"AAA": "nganh_tuy_chinh_1", "BBB": "nganh_tuy_chinh_2"}}

        run_full_market(config, delay_seconds=0)

        storage = Storage(db_path=db_path)
        completed = load_checkpoint(storage)
        # Chỉ 2 mã trong danh sách tùy chỉnh được xử lý (không phải 5 mã
        # từ FakeListing giả lập toàn thị trường trong fixture)
        assert completed == {"AAA", "BBB"}

        record = storage.get_latest("symbol_sector", "AAA")
        assert record["data"]["sector"] == "nganh_tuy_chinh_1"
        storage.close()


class TestNoDataLeakageAcrossRuns:
    """Mô phỏng ĐÚNG sự cố thực tế xảy ra 27/07/2026: chạy lần 1 với danh
    sách LỚN (giống quét toàn thị trường), sau đó chạy lần 2 với danh sách
    NHỎ HƠN, KHÁC HẲN (giống watchlist tùy chỉnh) — TRÊN CÙNG 1 storage
    (không xóa `./data` giữa 2 lần, đúng như tình huống thật của người
    dùng). Báo cáo tổng hợp của lần chạy 2 TUYỆT ĐỐI không được lẫn dữ
    liệu/ngành từ lần chạy 1.
    """

    def test_second_run_with_different_watchlist_does_not_leak_old_sectors(
        self, monkeypatch, no_sleep, config_with_db
    ):
        config, db_path = config_with_db

        # --- LẦN CHẠY 1: danh sách "toàn thị trường" giả lập, 5 mã, ngành vnstock ---
        fake_module_run1 = types.ModuleType("vnstock")

        class FakeEquity1:
            def ohlcv(self, interval, count):
                return pd.DataFrame({
                    "time": pd.bdate_range("2024-01-01", periods=260),
                    "open": [100.0] * 260, "high": [101.0] * 260,
                    "low": [99.0] * 260, "close": [100.0] * 260,
                    "volume": [100000] * 260,
                })

        class FakeMarket1:
            def equity(self, symbol):
                return FakeEquity1()

            def quote(self, symbol):
                return pd.DataFrame([{"symbol": symbol, "time": 1, "close_price": 100.0, "volume_accumulated": 1000}])

        class FakeListing1:
            def __init__(self, source=None):
                pass

            def symbols_by_industries(self):
                return pd.DataFrame({
                    "symbol": ["OLD1", "OLD2", "OLD3", "OLD4", "OLD5"],
                    "industry_code": ["1", "1", "2", "2", "3"],
                    "industry_name": ["Ngành Cũ A", "Ngành Cũ A", "Ngành Cũ B", "Ngành Cũ B", "Ngành Cũ C"],
                })

        fake_module_run1.Market = FakeMarket1
        fake_module_run1.Listing = FakeListing1
        monkeypatch.setitem(sys.modules, "vnstock", fake_module_run1)

        run_full_market(config, delay_seconds=0)  # config CHƯA có watchlist.symbols -> quét "toàn thị trường" giả lập

        storage_check1 = Storage(db_path=db_path)
        assert set(storage_check1.query_all_keys("symbol_sector")) == {"OLD1", "OLD2", "OLD3", "OLD4", "OLD5"}
        storage_check1.close()

        # --- LẦN CHẠY 2: watchlist TÙY CHỈNH, chỉ 2 mã, ngành HOÀN TOÀN khác ---
        config["watchlist"] = {"symbols": {"NEW1": "nganh_moi_x", "NEW2": "nganh_moi_y"}}

        fake_module_run2 = types.ModuleType("vnstock")

        class FakeEquity2:
            def ohlcv(self, interval, count):
                return pd.DataFrame({
                    "time": pd.bdate_range("2024-01-01", periods=260),
                    "open": [200.0] * 260, "high": [201.0] * 260,
                    "low": [199.0] * 260, "close": [200.0] * 260,
                    "volume": [50000] * 260,
                })

        class FakeMarket2:
            def equity(self, symbol):
                return FakeEquity2()

            def quote(self, symbol):
                return pd.DataFrame([{"symbol": symbol, "time": 1, "close_price": 200.0, "volume_accumulated": 500}])

        fake_module_run2.Market = FakeMarket2
        # KHÔNG cần Listing() ở lần 2 vì có watchlist.symbols tùy chỉnh -> không gọi fetch_symbol_sector_map()
        monkeypatch.setitem(sys.modules, "vnstock", fake_module_run2)

        run_full_market(config, delay_seconds=0, reset_checkpoint=True)

        storage = Storage(db_path=db_path)

        # 1. Báo cáo tín hiệu tổng hợp CHỈ được có 2 mã của lần chạy 2, KHÔNG lẫn 5 mã cũ
        signal_record = storage.get_latest("signal_summary_report", "latest")
        tong_so_ma = signal_record["data"]["tong_so_ma"]
        assert tong_so_ma == 2, f"Kỳ vọng đúng 2 mã, nhưng có {tong_so_ma} (rò rỉ dữ liệu cũ!)"

        # 2. Giai đoạn thị trường CHỈ được tổng hợp cho 2 ngành mới, KHÔNG có 3 ngành cũ
        assert storage.get_latest("market_regime_quant", "nganh_moi_x") is not None
        assert storage.get_latest("market_regime_quant", "nganh_moi_y") is not None
        assert storage.get_latest("market_regime_quant", "Ngành Cũ A") is None
        assert storage.get_latest("market_regime_quant", "Ngành Cũ B") is None
        assert storage.get_latest("market_regime_quant", "Ngành Cũ C") is None

        storage.close()


class TestComputeCapitalAllocationsForAllSymbols:
    def test_populates_both_allocation_types_when_data_available(self, config_with_db):
        config, db_path = config_with_db
        storage = Storage(db_path=db_path)

        n = 260
        closes = [30.0 + i * 0.05 for i in range(n)]  # xu hướng tăng nhẹ, đủ dữ liệu ATR/EMA200
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": closes, "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes], "close": closes,
            "volume": [100000] * n,
        })
        ohlcv_records = df.copy()
        ohlcv_records["date"] = ohlcv_records["date"].astype(str)
        storage.save("ohlcv_history", "AAA", {"records": ohlcv_records.to_dict("records")})
        storage.save("indicator_snapshot", "AAA", {
            "close": closes[-1], "ma20": closes[-1], "ema50": closes[-1],
            "ema100": closes[-1], "ema200": closes[-100],
            "price_above_ema200": True, "volume_ma_15": 100000, "volume_ma_20": 100000,
        })
        # Seed sẵn pattern_result để capital_allocator (đơn giản) có vùng entry
        storage.save("pattern_result", "AAA", {
            "segments": [{"low": 29.0, "high": 31.0, "amplitude_pct": 6.0}],
            "confidence": 0.7, "accumulation_high": 31.0, "effective_scan_months": 12,
        })
        storage.save("market_regime_quant", "test_sector", {
            "trang_thai": "UPTREND", "do_tin_cay": "CAO", "macro_score": 0.5,
            "breadth_pct": 70.0, "breadth_theo_nhom": "test_sector",
            "canh_bao": [], "reasoning": [],
        })
        storage.close()

        storage = Storage(db_path=db_path)
        compute_capital_allocations_for_all_symbols(storage, {"AAA": "test_sector"}, config)

        simple_result = storage.get_latest("allocation_recommendation", "AAA")
        v2_result = storage.get_latest("capital_allocation_v2", "AAA")

        assert simple_result is not None, "Phân bổ vốn ĐƠN GIẢN vẫn trống dù đã có đủ dữ liệu!"
        assert v2_result is not None, "Phân bổ vốn ATR14 CHI TIẾT vẫn trống dù đã có đủ dữ liệu!"
        storage.close()

    def test_skips_simple_allocation_gracefully_without_pattern_result(self, config_with_db):
        """Không có pattern_result -> phân bổ vốn ĐƠN GIẢN bị bỏ qua (đúng
        thiết kế), nhưng KHÔNG được làm hỏng việc tính ATR14 chi tiết.
        """
        config, db_path = config_with_db
        storage = Storage(db_path=db_path)

        n = 260
        closes = [30.0 + i * 0.05 for i in range(n)]
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": closes, "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes], "close": closes,
            "volume": [100000] * n,
        })
        ohlcv_records = df.copy()
        ohlcv_records["date"] = ohlcv_records["date"].astype(str)
        storage.save("ohlcv_history", "BBB", {"records": ohlcv_records.to_dict("records")})
        storage.save("indicator_snapshot", "BBB", {
            "close": closes[-1], "ma20": closes[-1], "ema50": closes[-1],
            "ema100": closes[-1], "ema200": closes[-100],
            "price_above_ema200": True, "volume_ma_15": 100000, "volume_ma_20": 100000,
        })
        storage.save("market_regime_quant", "test_sector2", {
            "trang_thai": "UPTREND", "do_tin_cay": "CAO", "macro_score": 0.5,
            "breadth_pct": 70.0, "breadth_theo_nhom": "test_sector2",
            "canh_bao": [], "reasoning": [],
        })
        storage.close()

        storage = Storage(db_path=db_path)
        compute_capital_allocations_for_all_symbols(storage, {"BBB": "test_sector2"}, config)

        assert storage.get_latest("allocation_recommendation", "BBB") is None
        assert storage.get_latest("capital_allocation_v2", "BBB") is not None
        storage.close()


class TestStockCharacterIntegration:
    """Xác nhận tính cách giao dịch (core/stock_character_classifier.py)
    được tính VÀ áp dụng điều chỉnh vào cả tín hiệu mua/bán lẫn phân bổ
    vốn khi chạy qua run_full_market.py — không chỉ qua main.py."""

    def test_stock_character_computed_and_saved(self, config_with_db):
        config, db_path = config_with_db
        storage = Storage(db_path=db_path)

        n = 600
        # Dữ liệu tăng đều để đủ 600 phiên (>MIN_HISTORY_FOR_FULL_CONFIDENCE)
        closes = [30.0 + i * 0.01 for i in range(n)]
        df = pd.DataFrame({
            "date": pd.bdate_range("2022-01-01", periods=n),
            "open": closes, "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes], "close": closes,
            "volume": [100000] * n,
        })
        ohlcv_records = df.copy()
        ohlcv_records["date"] = ohlcv_records["date"].astype(str)
        storage.save("ohlcv_history", "AAA", {"records": ohlcv_records.to_dict("records")})
        storage.save("indicator_snapshot", "AAA", {
            "close": closes[-1], "ma20": closes[-1], "ema50": closes[-1],
            "ema100": closes[-1], "ema200": closes[-100],
            "price_above_ema200": True, "volume_ma_15": 100000, "volume_ma_20": 100000,
        })
        storage.save("pattern_result", "AAA", {
            "segments": [{"low": 29.0, "high": 31.0, "amplitude_pct": 6.0}],
            "confidence": 0.7, "accumulation_high": 31.0, "effective_scan_months": 12,
        })
        storage.save("market_regime_quant", "test_sector", {
            "trang_thai": "UPTREND", "do_tin_cay": "CAO", "macro_score": 0.5,
            "breadth_pct": 70.0, "breadth_theo_nhom": "test_sector",
            "canh_bao": [], "reasoning": [],
        })
        storage.close()

        from run_full_market import compute_capital_allocations_for_all_symbols, compute_stock_signals_for_all_symbols

        storage = Storage(db_path=db_path)
        compute_stock_signals_for_all_symbols(storage, {"AAA": "test_sector"})
        compute_capital_allocations_for_all_symbols(storage, {"AAA": "test_sector"}, config)

        character_record = storage.get_latest("stock_character", "AAA")
        assert character_record is not None
        assert "nhan_tinh_cach" in character_record["data"]

        signal_record = storage.get_latest("stock_signal", "AAA")
        assert signal_record is not None
        assert "nhan_tinh_cach" in signal_record["data"]  # đã được điều chỉnh, gắn thêm nhãn

        storage.close()
