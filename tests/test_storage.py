"""
Unit test cho core/storage.py

Dùng SQLite in-memory (":memory:") cho mọi test backend SQLite — không
tạo file trên đĩa.

Backend PostgreSQL/Supabase được test bằng cách GIẢ LẬP kết nối
`psycopg2` (không có server Postgres thật trong môi trường test/sandbox)
— xác nhận đúng: nhận diện connection string, câu lệnh SQL sinh ra đúng
cú pháp Postgres (SERIAL, %s, RETURNING id), luồng gọi commit/fetchone
đúng thứ tự. Việc kết nối THẬT tới Supabase cần người dùng tự xác nhận
bằng script `test_supabase_connection.py` (xem hướng dẫn kèm theo).
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from core.storage import Storage, StorageError, _is_postgres_connection_string


@pytest.fixture
def storage():
    s = Storage(db_path=":memory:")
    yield s
    s.close()


# ==============================================================================
# Test: save + get_latest
# ==============================================================================

class TestSaveAndGetLatest:
    def test_roundtrip_basic_dict(self, storage):
        storage.save("market_regime", "banking", {"regime": "uptrend", "confidence": 0.8})
        result = storage.get_latest("market_regime", "banking")

        assert result is not None
        assert result["data"] == {"regime": "uptrend", "confidence": 0.8}

    def test_returns_none_when_no_data(self, storage):
        result = storage.get_latest("market_regime", "khong_ton_tai")
        assert result is None

    def test_latest_returns_most_recent_of_multiple_saves(self, storage):
        storage.save(
            "market_regime", "banking", {"regime": "sideway"},
            timestamp=datetime(2026, 1, 1),
        )
        storage.save(
            "market_regime", "banking", {"regime": "uptrend"},
            timestamp=datetime(2026, 1, 5),
        )
        result = storage.get_latest("market_regime", "banking")
        assert result["data"]["regime"] == "uptrend"

    def test_different_keys_do_not_interfere(self, storage):
        storage.save("market_regime", "banking", {"regime": "uptrend"})
        storage.save("market_regime", "real_estate", {"regime": "downtrend"})

        banking = storage.get_latest("market_regime", "banking")
        real_estate = storage.get_latest("market_regime", "real_estate")

        assert banking["data"]["regime"] == "uptrend"
        assert real_estate["data"]["regime"] == "downtrend"

    def test_different_categories_do_not_interfere(self, storage):
        storage.save("market_regime", "HPG", {"regime": "uptrend"})
        storage.save("pattern_result", "HPG", {"confidence": 0.7})

        regime = storage.get_latest("market_regime", "HPG")
        pattern = storage.get_latest("pattern_result", "HPG")

        assert "regime" in regime["data"]
        assert "confidence" in pattern["data"]

    def test_raises_for_empty_category_or_key(self, storage):
        with pytest.raises(StorageError):
            storage.save("", "HPG", {"a": 1})
        with pytest.raises(StorageError):
            storage.save("category", "", {"a": 1})


# ==============================================================================
# Test: get_history
# ==============================================================================

class TestGetHistory:
    def test_returns_records_newest_first(self, storage):
        storage.save("indicator_snapshot", "HPG", {"close": 100}, timestamp=datetime(2026, 1, 1))
        storage.save("indicator_snapshot", "HPG", {"close": 105}, timestamp=datetime(2026, 1, 2))
        storage.save("indicator_snapshot", "HPG", {"close": 110}, timestamp=datetime(2026, 1, 3))

        history = storage.get_history("indicator_snapshot", "HPG")
        closes = [h["data"]["close"] for h in history]
        assert closes == [110, 105, 100]

    def test_respects_limit(self, storage):
        for i in range(10):
            storage.save(
                "indicator_snapshot", "HPG", {"close": i},
                timestamp=datetime(2026, 1, 1) + timedelta(days=i),
            )
        history = storage.get_history("indicator_snapshot", "HPG", limit=3)
        assert len(history) == 3


# ==============================================================================
# Test: query_all_keys / query_all_categories
# ==============================================================================

class TestQueryKeysAndCategories:
    def test_query_all_keys_returns_distinct_symbols(self, storage):
        storage.save("ohlcv", "HPG", {"close": 100})
        storage.save("ohlcv", "VNM", {"close": 200})
        storage.save("ohlcv", "HPG", {"close": 101})  # trùng key, không nhân đôi

        keys = storage.query_all_keys("ohlcv")
        assert sorted(keys) == ["HPG", "VNM"]

    def test_query_all_categories(self, storage):
        storage.save("ohlcv", "HPG", {"close": 100})
        storage.save("market_regime", "banking", {"regime": "uptrend"})

        categories = storage.query_all_categories()
        assert set(categories) == {"ohlcv", "market_regime"}


# ==============================================================================
# Test: dọn dẹp dữ liệu cũ
# ==============================================================================

class TestDeleteOlderThan:
    def test_deletes_only_records_before_cutoff(self, storage):
        storage.save("ohlcv", "HPG", {"close": 100}, timestamp=datetime(2025, 1, 1))
        storage.save("ohlcv", "HPG", {"close": 200}, timestamp=datetime(2026, 6, 1))

        deleted = storage.delete_older_than("ohlcv", cutoff=datetime(2026, 1, 1))

        assert deleted == 1
        remaining = storage.get_history("ohlcv", "HPG")
        assert len(remaining) == 1
        assert remaining[0]["data"]["close"] == 200


class TestDeleteKey:
    def test_deletes_all_records_for_key(self, storage):
        storage.save("chart_annotation", "ann-1", {"text": "Sự kiện A"})
        storage.save("chart_annotation", "ann-1", {"text": "Sự kiện A (sửa)"})
        storage.save("chart_annotation", "ann-2", {"text": "Sự kiện B"})

        deleted = storage.delete_key("chart_annotation", "ann-1")

        assert deleted == 2  # cả 2 bản ghi cũ của ann-1 đều bị xóa
        assert storage.get_latest("chart_annotation", "ann-1") is None
        assert storage.get_latest("chart_annotation", "ann-2") is not None

    def test_returns_zero_when_key_not_found(self, storage):
        deleted = storage.delete_key("chart_annotation", "khong_ton_tai")
        assert deleted == 0


# ==============================================================================
# Test: dữ liệu chứa datetime/pandas.Timestamp phải serialize được
# ==============================================================================

class TestSerializationOfSpecialTypes:
    def test_handles_datetime_and_timestamp_inside_data(self, storage):
        data = {
            "as_of": datetime(2026, 1, 1, 10, 30),
            "scan_start_date": pd.Timestamp("2026-01-01"),
            "segments": [{"start_date": pd.Timestamp("2026-01-01"), "amplitude_pct": 5.0}],
        }
        storage.save("pattern_result", "HPG", data)
        result = storage.get_latest("pattern_result", "HPG")

        assert result is not None
        assert "2026-01-01" in result["data"]["as_of"]
        assert "2026-01-01" in result["data"]["scan_start_date"]


# ==============================================================================
# Test: context manager
# ==============================================================================

class TestContextManager:
    def test_can_be_used_as_context_manager(self):
        with Storage(db_path=":memory:") as s:
            s.save("test", "key1", {"a": 1})
            result = s.get_latest("test", "key1")
            assert result["data"] == {"a": 1}


# ==============================================================================
# Test: nhận diện backend PostgreSQL/Supabase qua connection string
# ==============================================================================

class TestIsPostgresConnectionString:
    def test_recognizes_postgresql_prefix(self):
        assert _is_postgres_connection_string("postgresql://user:pass@host:5432/db") is True

    def test_recognizes_postgres_prefix(self):
        assert _is_postgres_connection_string("postgres://user:pass@host:5432/db") is True

    def test_local_sqlite_path_is_not_postgres(self):
        assert _is_postgres_connection_string("./data/pm_ck.db") is False

    def test_memory_sqlite_is_not_postgres(self):
        assert _is_postgres_connection_string(":memory:") is False


# ==============================================================================
# Test: backend PostgreSQL/Supabase — GIẢ LẬP kết nối (không có server thật)
# ==============================================================================

def _make_fake_psycopg2_module():
    """Module `psycopg2` giả lập — trả về connection/cursor giả để kiểm
    tra ĐÚNG câu lệnh SQL và luồng gọi hàm, không cần server Postgres thật.
    """
    fake_module = types.ModuleType("psycopg2")
    fake_extras = types.ModuleType("psycopg2.extras")

    class FakeRealDictCursor:
        pass

    fake_extras.RealDictCursor = FakeRealDictCursor
    fake_module.extras = fake_extras

    return fake_module, fake_extras


class TestPostgresBackend:
    def test_connects_with_correct_connection_string_and_autocommit_false(self, monkeypatch):
        fake_module, _ = _make_fake_psycopg2_module()
        fake_conn = MagicMock()
        fake_cursor = MagicMock()
        fake_conn.cursor.return_value = fake_cursor
        fake_module.connect = MagicMock(return_value=fake_conn)

        monkeypatch.setitem(sys.modules, "psycopg2", fake_module)
        monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_module.extras)

        conn_str = "postgresql://user:pass@aws-region.pooler.supabase.com:5432/postgres"
        storage = Storage(db_path=conn_str)

        fake_module.connect.assert_called_once_with(conn_str)
        assert fake_conn.autocommit is False
        storage.close()

    def test_create_tables_uses_postgres_syntax_not_sqlite(self, monkeypatch):
        fake_module, _ = _make_fake_psycopg2_module()
        fake_conn = MagicMock()
        fake_cursor = MagicMock()
        fake_conn.cursor.return_value = fake_cursor
        fake_module.connect = MagicMock(return_value=fake_conn)

        monkeypatch.setitem(sys.modules, "psycopg2", fake_module)
        monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_module.extras)

        Storage(db_path="postgresql://x:y@host:5432/db")

        executed_sql = [c.args[0] for c in fake_cursor.execute.call_args_list]
        create_table_sql = next(sql for sql in executed_sql if "CREATE TABLE" in sql)

        assert "SERIAL PRIMARY KEY" in create_table_sql
        assert "AUTOINCREMENT" not in create_table_sql

    def test_save_uses_returning_id_and_fetchone(self, monkeypatch):
        fake_module, _ = _make_fake_psycopg2_module()
        fake_conn = MagicMock()
        fake_cursor = MagicMock()
        fake_cursor.fetchone.return_value = {"id": 42}
        fake_conn.cursor.return_value = fake_cursor
        fake_module.connect = MagicMock(return_value=fake_conn)

        monkeypatch.setitem(sys.modules, "psycopg2", fake_module)
        monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_module.extras)

        storage = Storage(db_path="postgresql://x:y@host:5432/db")
        new_id = storage.save("test_category", "test_key", {"gia": 100})

        assert new_id == 42
        insert_call = next(
            c for c in fake_cursor.execute.call_args_list if "INSERT INTO records" in c.args[0]
        )
        assert "RETURNING id" in insert_call.args[0]
        assert "%s" in insert_call.args[0]
        assert insert_call.args[1][0] == "test_category"
        assert insert_call.args[1][1] == "test_key"

    def test_get_latest_uses_percent_s_placeholders(self, monkeypatch):
        fake_module, _ = _make_fake_psycopg2_module()
        fake_conn = MagicMock()
        fake_cursor = MagicMock()
        fake_cursor.fetchone.return_value = {
            "timestamp": "2026-01-01T00:00:00", "data": '{"gia": 100}',
        }
        fake_conn.cursor.return_value = fake_cursor
        fake_module.connect = MagicMock(return_value=fake_conn)

        monkeypatch.setitem(sys.modules, "psycopg2", fake_module)
        monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_module.extras)

        storage = Storage(db_path="postgresql://x:y@host:5432/db")
        result = storage.get_latest("test_category", "test_key")

        assert result == {"timestamp": "2026-01-01T00:00:00", "data": {"gia": 100}}
        select_call = next(
            c for c in fake_cursor.execute.call_args_list if "SELECT timestamp, data" in c.args[0]
        )
        assert "%s" in select_call.args[0]
        assert "?" not in select_call.args[0]

    def test_raises_storage_error_when_connection_fails(self, monkeypatch):
        fake_module, _ = _make_fake_psycopg2_module()
        fake_module.connect = MagicMock(side_effect=Exception("connection refused"))

        monkeypatch.setitem(sys.modules, "psycopg2", fake_module)
        monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_module.extras)

        with pytest.raises(StorageError, match="Không kết nối được"):
            Storage(db_path="postgresql://x:y@host:5432/db")

    def test_raises_storage_error_when_psycopg2_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psycopg2", None)

        with pytest.raises(StorageError, match="psycopg2-binary"):
            Storage(db_path="postgresql://x:y@host:5432/db")


# ==============================================================================
# Test: _row_to_dict — dự phòng truy cập theo vị trí cột khi truy cập
# theo tên cột thất bại (rà soát sau sự cố "IndexError: tuple index out
# of range" thực tế 28/07/2026 — nguyên nhân gốc chưa xác định chắc chắn,
# nhưng đây là lớp bảo vệ hợp lý bất kể nguyên nhân).
# ==============================================================================

class TestRowToDictFallback:
    def test_falls_back_to_positional_access_when_name_access_fails(self):
        class FakeRowRaisingOnStringKey(tuple):
            """Giả lập 1 row KHÔNG hỗ trợ truy cập theo tên cột (chỉ hỗ
            trợ theo vị trí) — mô phỏng tình huống row_factory không như
            mong đợi."""
            def __getitem__(self, item):
                if isinstance(item, str):
                    raise IndexError("No item with that key")
                return tuple.__getitem__(self, item)

        fake_row = FakeRowRaisingOnStringKey(("2026-01-01T00:00:00", '{"gia": 100}'))
        result = Storage._row_to_dict(fake_row)

        assert result == {"timestamp": "2026-01-01T00:00:00", "data": {"gia": 100}}

    def test_normal_dict_like_row_still_works(self):
        fake_row = {"timestamp": "2026-01-01T00:00:00", "data": '{"gia": 100}'}
        result = Storage._row_to_dict(fake_row)
        assert result == {"timestamp": "2026-01-01T00:00:00", "data": {"gia": 100}}

    def test_end_to_end_get_latest_still_works_normally(self, storage):
        storage.save("test_category", "test_key", {"gia": 100})
        result = storage.get_latest("test_category", "test_key")
        assert result["data"] == {"gia": 100}
