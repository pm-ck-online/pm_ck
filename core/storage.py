"""
storage.py
===========
[Bổ sung — cần thiết cho dashboard đọc dữ liệu ở Giai đoạn 5]
[Nâng cấp 27/07/2026 — hỗ trợ THÊM backend PostgreSQL/Supabase để dùng
 CHUNG dữ liệu giữa nhiều máy tính, song song với SQLite cục bộ ban đầu]

Lưu trữ dữ liệu đã thu thập/tính toán từ các module khác (OHLCV, chỉ báo,
kết quả pattern_detector, market_regime, khuyến nghị capital_allocator,
snapshot paper_portfolio...) để `dashboard/app.py` đọc lại mà KHÔNG cần
gọi trực tiếp `data_collector` hay tính toán lại — tránh trùng lặp việc
gọi API và giữ dashboard đơn giản, chỉ đọc.

THIẾT KẾ: một bảng LƯU TRỮ TỔNG QUÁT (generic key-value + JSON) thay vì
một bảng riêng cho từng loại dữ liệu — vì output của các module khác nhau
(indicators, pattern_detector, market_regime_detector...) có cấu trúc
dict lồng nhau khác nhau và có thể thay đổi theo thời gian. Mỗi bản ghi
được xác định bởi `(category, key)`, ví dụ:
    - category="ohlcv", key="HPG"
    - category="market_regime", key="banking"
    - category="portfolio_snapshot", key="default"

HAI BACKEND — TỰ ĐỘNG NHẬN DIỆN qua `db_path`:
    - SQLite (mặc định, dùng file cục bộ): `db_path="./data/pm_ck.db"` hoặc
      `db_path=":memory:"` (test).
    - PostgreSQL/Supabase (dùng CHUNG nhiều máy): `db_path` là connection
      string bắt đầu bằng "postgresql://" hoặc "postgres://" — ví dụ lấy
      từ Supabase Dashboard -> Project Settings -> Database -> Connection
      string.

Toàn bộ API công khai (`save`, `get_latest`, `get_history`,
`query_all_keys`, `query_all_categories`, `delete_older_than`,
`delete_key`, `close`) GIỮ NGUYÊN HỆT giữa 2 backend — mọi code gọi
`Storage` (main.py, run_full_market.py, dashboard/app.py) KHÔNG cần sửa
gì khi đổi giữa 2 backend, chỉ cần đổi giá trị `storage.path` trong
config.yaml.

Module này CHỈ lưu và đọc lại — không tính toán nghiệp vụ.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Optional


def _json_default(obj):
    """Chuyển đổi các kiểu dữ liệu không tự nhiên JSON hóa được (datetime,
    date, pandas.Timestamp...) sang chuỗi ISO — dùng làm `default=` cho
    `json.dumps`.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):  # bao gồm pandas.Timestamp
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy scalar (int64, float64...)
        return obj.item()
    return str(obj)


class StorageError(Exception):
    """Lỗi phát sinh khi thao tác với storage."""


def _is_postgres_connection_string(db_path: str) -> bool:
    """Nhận diện `db_path` có phải connection string PostgreSQL/Supabase
    hay không (thay vì đường dẫn file SQLite cục bộ)."""
    return db_path.startswith("postgresql://") or db_path.startswith("postgres://")


# Nhận diện lỗi MẤT KẾT NỐI Supabase — dùng CHUNG cho mọi nơi cần tự kết
# nối lại khi lưu/đọc dữ liệu trong 1 tiến trình chạy dài (VD run_full_
# market.py, main.run_market_regime_history_step). Ban đầu chỉ khai báo
# riêng ở run_full_market.py (04/08/2026, cho bước lưu checkpoint) — đưa
# về đây (25/08/2026) để dùng lại được ở nhiều nơi mà không phải import
# chéo giữa main.py/run_full_market.py.
CONNECTION_ERROR_KEYWORDS = (
    "connection already closed", "connection is closed",
    "server closed the connection", "could not connect", "connection not open",
)


def is_connection_error(exc: BaseException) -> bool:
    """Nhận diện lỗi MẤT KẾT NỐI Supabase qua nội dung thông báo lỗi (thư
    viện psycopg2 không cung cấp 1 class ngoại lệ riêng biệt để bắt theo
    kiểu cho mọi trường hợp mất kết nối)."""
    text = str(exc).lower()
    return any(keyword in text for keyword in CONNECTION_ERROR_KEYWORDS)


class Storage:
    """Lớp lưu trữ chung cho toàn hệ thống — tự động chọn SQLite (file cục
    bộ) hoặc PostgreSQL/Supabase (dùng chung nhiều máy) dựa trên `db_path`.

    Dùng `db_path=":memory:"` để chạy trong bộ nhớ (phù hợp cho unit
    test — không tạo file trên đĩa, luôn dùng SQLite).
    """

    def __init__(self, db_path: str = "./data/pm_ck.db"):
        self.db_path = db_path
        self._is_postgres = _is_postgres_connection_string(db_path)

        if self._is_postgres:
            self._init_postgres(db_path)
        else:
            self._init_sqlite(db_path)

        self._create_tables()

    # --------------------------------------------------------------------
    # Khởi tạo kết nối theo từng backend
    # --------------------------------------------------------------------
    def _init_sqlite(self, db_path: str) -> None:
        import sqlite3

        if db_path != ":memory:":
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        # check_same_thread=False: cần thiết vì Streamlit (dashboard/app.py)
        # có thể chạy script ở một thread khác với thread đã tạo kết nối
        # khi dùng chung với @st.cache_resource. Truy cập vẫn tuần tự
        # (Streamlit không ghi đồng thời đa luồng vào cùng 1 session), nên
        # an toàn cho quy mô sử dụng của dự án này.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._placeholder = "?"

    def _init_postgres(self, connection_string: str) -> None:
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError as exc:
            raise StorageError(
                "Cần cài đặt 'psycopg2-binary' để dùng PostgreSQL/Supabase: "
                "pip install psycopg2-binary"
            ) from exc

        try:
            self._conn = psycopg2.connect(connection_string)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(
                f"Không kết nối được tới PostgreSQL/Supabase. Kiểm tra lại "
                f"connection string (Project Settings -> Database trên "
                f"Supabase Dashboard). Lỗi gốc: {exc}"
            ) from exc

        self._conn.autocommit = False
        self._cursor_factory = psycopg2.extras.RealDictCursor
        self._placeholder = "%s"

    # --------------------------------------------------------------------
    # Tạo bảng (cú pháp khác nhau đôi chút giữa SQLite/Postgres)
    # --------------------------------------------------------------------
    def _create_tables(self) -> None:
        if self._is_postgres:
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id SERIAL PRIMARY KEY,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL
                )
                """
            )
        else:
            self._execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL
                )
                """
            )
        self._execute(
            "CREATE INDEX IF NOT EXISTS idx_category_key ON records (category, key)"
        )
        self._commit()

    # --------------------------------------------------------------------
    # Lớp bọc thực thi SQL — che giấu khác biệt cursor giữa 2 backend
    # --------------------------------------------------------------------
    def _execute(self, sql: str, params: tuple = ()):
        if self._is_postgres:
            cursor = self._conn.cursor(cursor_factory=self._cursor_factory)
            cursor.execute(sql, params)
            return cursor
        return self._conn.execute(sql, params)

    def _commit(self) -> None:
        self._conn.commit()

    def _voi_ket_noi_lai(self, thao_tac):
        """Chạy `thao_tac` (hàm KHÔNG tham số, gồm toàn bộ execute+fetch+
        commit của 1 thao tác đọc/ghi); nếu Supabase tự đóng kết nối giữa
        chừng, TỰ ĐỘNG mở lại kết nối rồi chạy lại TOÀN BỘ `thao_tac` từ
        đầu đúng 1 lần (bổ sung 25/08/2026 — dùng CHUNG cho MỌI phương
        thức đọc/ghi, thay vì phải vá riêng lẻ từng nơi GỌI Storage mỗi
        khi phát hiện thêm 1 chỗ bị lỗi mất kết nối giữa chừng, như đã
        từng làm với checkpoint và chuỗi giai đoạn theo ngành).

        Phải retry LẠI TỪ ĐẦU cả execute lẫn commit (không chỉ riêng 1
        trong 2) — vì cả 2 cùng thuộc 1 transaction: mất kết nối giữa
        execute và commit coi như TOÀN BỘ thao tác chưa từng xảy ra, kết
        nối mới không hề "nhớ" phần đã execute dở trên kết nối cũ.

        Chỉ áp dụng cho backend Postgres — SQLite cục bộ không có khái
        niệm server tự đóng kết nối giữa chừng.
        """
        if not self._is_postgres:
            return thao_tac()
        try:
            return thao_tac()
        except Exception as exc:  # noqa: BLE001
            if not is_connection_error(exc):
                raise
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._init_postgres(self.db_path)
            return thao_tac()

    # --------------------------------------------------------------------
    # Ghi dữ liệu
    # --------------------------------------------------------------------
    def save(
        self,
        category: str,
        key: str,
        data: dict,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """Lưu một bản ghi mới cho `(category, key)`. Mỗi lần gọi tạo một
        bản ghi MỚI (append-only) — không ghi đè bản ghi cũ, để giữ lại
        lịch sử đầy đủ (ví dụ lịch sử giai đoạn thị trường qua từng ngày).

        Trả về `id` của bản ghi vừa tạo.
        """
        if not category or not key:
            raise StorageError("category và key không được để trống.")

        timestamp = timestamp or datetime.now()
        serialized = json.dumps(data, default=_json_default, ensure_ascii=False)

        def _lam():
            if self._is_postgres:
                cursor = self._execute(
                    "INSERT INTO records (category, key, timestamp, data) "
                    "VALUES (%s, %s, %s, %s) RETURNING id",
                    (category, key, timestamp.isoformat(), serialized),
                )
                new_id = cursor.fetchone()["id"]
                self._commit()
                return new_id

            cursor = self._execute(
                "INSERT INTO records (category, key, timestamp, data) VALUES (?, ?, ?, ?)",
                (category, key, timestamp.isoformat(), serialized),
            )
            self._commit()
            return cursor.lastrowid

        return self._voi_ket_noi_lai(_lam)

    # --------------------------------------------------------------------
    # Đọc dữ liệu
    # --------------------------------------------------------------------
    @staticmethod
    def _row_to_dict(row) -> dict:
        """Chuyển 1 row (SELECT timestamp, data ...) thành dict — ưu tiên
        truy cập theo TÊN CỘT (sqlite3.Row / RealDictCursor), nhưng có
        DỰ PHÒNG truy cập theo VỊ TRÍ cột (row[0]=timestamp, row[1]=data)
        nếu truy cập theo tên thất bại vì bất kỳ lý do gì — câu lệnh
        SELECT luôn cố định đúng thứ tự "timestamp, data" nên vị trí cột
        luôn đáng tin cậy làm phương án dự phòng.
        """
        try:
            timestamp, data = row["timestamp"], row["data"]
        except (KeyError, IndexError, TypeError):
            timestamp, data = row[0], row[1]
        return {"timestamp": timestamp, "data": json.loads(data)}

    def get_latest(self, category: str, key: str) -> Optional[dict]:
        """Lấy bản ghi MỚI NHẤT cho `(category, key)`. Trả về None nếu
        chưa có dữ liệu nào.
        """
        def _lam():
            p = self._placeholder
            cursor = self._execute(
                f"""
                SELECT timestamp, data FROM records
                WHERE category = {p} AND key = {p}
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (category, key),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_dict(row)

        return self._voi_ket_noi_lai(_lam)

    def get_latest_many(self, category: str, keys: list[str]) -> dict[str, dict]:
        """Lấy bản ghi MỚI NHẤT cho NHIỀU key cùng lúc, trong ĐÚNG 1 câu
        truy vấn SQL — thay vì gọi `get_latest()` lặp lại cho từng key
        (mỗi lần là 1 lượt round-trip mạng riêng tới Supabase, rất chậm
        khi watchlist có nhiều mã vì độ trễ mạng cộng dồn).

        Cách làm: lấy VỀ TẤT CẢ bản ghi khớp (category, key IN keys),
        sắp xếp key rồi timestamp giảm dần — sau đó chỉ giữ lại bản ghi
        ĐẦU TIÊN gặp cho mỗi key (chính là bản mới nhất, vì đã sắp xếp
        timestamp giảm dần). Cách này hoạt động giống hệt trên cả SQLite
        và PostgreSQL, không cần cú pháp DISTINCT ON/window function
        riêng cho từng backend.

        Trả về dict {key: {"timestamp":..., "data": {...}}} — key nào
        không có dữ liệu sẽ KHÔNG xuất hiện trong dict trả về.
        """
        if not keys:
            return {}

        def _lam():
            p = self._placeholder
            placeholders = ", ".join([p] * len(keys))
            cursor = self._execute(
                f"""
                SELECT key, timestamp, data FROM records
                WHERE category = {p} AND key IN ({placeholders})
                ORDER BY key, timestamp DESC, id DESC
                """,
                (category, *keys),
            )
            rows = cursor.fetchall()

            result: dict[str, dict] = {}
            for row in rows:
                try:
                    key, timestamp, data = row["key"], row["timestamp"], row["data"]
                except (KeyError, IndexError, TypeError):
                    key, timestamp, data = row[0], row[1], row[2]
                if key not in result:  # chỉ giữ bản ĐẦU TIÊN gặp = mới nhất
                    result[key] = {"timestamp": timestamp, "data": json.loads(data)}
            return result

        return self._voi_ket_noi_lai(_lam)

    def get_history(
        self, category: str, key: str, limit: int = 100
    ) -> list[dict]:
        """Lấy lịch sử các bản ghi cho `(category, key)`, mới nhất trước,
        giới hạn tối đa `limit` bản ghi.
        """
        def _lam():
            p = self._placeholder
            cursor = self._execute(
                f"""
                SELECT timestamp, data FROM records
                WHERE category = {p} AND key = {p}
                ORDER BY timestamp DESC, id DESC
                LIMIT {p}
                """,
                (category, key, limit),
            )
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

        return self._voi_ket_noi_lai(_lam)

    def query_all_keys(self, category: str) -> list[str]:
        """Trả về danh sách các `key` khác nhau (distinct) đã từng được
        lưu trong một `category` — ví dụ: tất cả mã cổ phiếu đã lưu OHLCV.
        """
        def _lam():
            p = self._placeholder
            cursor = self._execute(
                f"SELECT DISTINCT key FROM records WHERE category = {p} ORDER BY key",
                (category,),
            )
            return [row["key"] for row in cursor.fetchall()]

        return self._voi_ket_noi_lai(_lam)

    def query_all_categories(self) -> list[str]:
        """Trả về danh sách tất cả các `category` hiện có trong storage."""
        def _lam():
            cursor = self._execute(
                "SELECT DISTINCT category FROM records ORDER BY category"
            )
            return [row["category"] for row in cursor.fetchall()]

        return self._voi_ket_noi_lai(_lam)

    # --------------------------------------------------------------------
    # Dọn dẹp dữ liệu cũ
    # --------------------------------------------------------------------
    def delete_older_than(self, category: str, cutoff: datetime) -> int:
        """Xóa các bản ghi của `category` có timestamp CŨ HƠN `cutoff`.
        Trả về số bản ghi đã xóa.
        """
        def _lam():
            p = self._placeholder
            cursor = self._execute(
                f"DELETE FROM records WHERE category = {p} AND timestamp < {p}",
                (category, cutoff.isoformat()),
            )
            self._commit()
            return cursor.rowcount

        return self._voi_ket_noi_lai(_lam)

    def delete_key(self, category: str, key: str) -> int:
        """Xóa TOÀN BỘ bản ghi (mọi lịch sử) của một `(category, key)` cụ
        thể — dùng khi cần xóa hẳn 1 mục (vd. xóa 1 chú thích biểu đồ, 1
        giao dịch...). Trả về số bản ghi đã xóa.
        """
        def _lam():
            p = self._placeholder
            cursor = self._execute(
                f"DELETE FROM records WHERE category = {p} AND key = {p}",
                (category, key),
            )
            self._commit()
            return cursor.rowcount

        return self._voi_ket_noi_lai(_lam)

    # --------------------------------------------------------------------
    # Quản lý kết nối
    # --------------------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
