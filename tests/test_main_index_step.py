"""Test cho `main.run_index_step()` — bổ sung 27/08/2026 theo yêu cầu
người dùng thêm mã VN30 (và các chỉ số khác) vào mục "🎭 Tính cách giao
dịch từng mã" / "🧮 Cổ phiếu dài hạn" trên dashboard.

Cả 2 mục đó đọc dữ liệu GENERIC qua `storage.query_all_keys(...)`, không
lọc riêng theo `watchlist.symbols` — nên chỉ cần `run_index_step()` cũng
lưu category `stock_character` (giống cổ phiếu thường) là chỉ số sẽ TỰ
ĐỘNG xuất hiện trên dashboard, không cần sửa gì thêm ở `dashboard/app.py`.
"""

from __future__ import annotations

from core.data_collector import DataCollector, MockDataSource
from core.storage import Storage
from main import run_index_step, run_long_term_screener_step


class TestRunIndexStepTinhCachGiaoDich:
    def test_luu_ca_indicator_snapshot_lan_stock_character(self):
        storage = Storage(db_path=":memory:")
        collector = DataCollector(MockDataSource())

        snapshot = run_index_step(collector, storage, "VN30", config={})

        assert snapshot is not None
        assert storage.get_latest("indicator_snapshot", "VN30") is not None
        # HÀNH VI MỚI: trước đây chỉ số KHÔNG có "stock_character" (chỉ
        # cổ phiếu thường mới có) -> mục "Tính cách giao dịch từng mã"
        # (đọc storage.query_all_keys("stock_character")) không bao giờ
        # hiện chỉ số. Giờ phải có, để VN30 tự động xuất hiện ở đó.
        character_record = storage.get_latest("stock_character", "VN30")
        assert character_record is not None
        assert character_record["data"].get("nhan_tinh_cach") is not None
        storage.close()

    def test_loi_lay_du_lieu_khong_luu_gi_va_tra_ve_none(self):
        """Không lấy được OHLCV chỉ số (VD lỗi mạng) -> không nên có
        stock_character "ma" cho mã đó (không có dữ liệu để tính)."""
        class _ThatBaiSource(MockDataSource):
            def fetch_ohlcv(self, symbol, timeframe="day"):
                raise RuntimeError("giả lập lỗi mạng")

        storage = Storage(db_path=":memory:")
        collector = DataCollector(_ThatBaiSource())

        ket_qua = run_index_step(collector, storage, "VN30", config={})

        assert ket_qua is None
        assert storage.get_latest("stock_character", "VN30") is None
        storage.close()


class TestRunLongTermScreenerStepChoChiSo:
    def test_chi_so_cung_duoc_tinh_bo_loc_dai_han(self):
        """`run_long_term_screener_step()` hoàn toàn generic theo
        symbol_sector_map — gọi với nhãn "index" cho 1 chỉ số đã có
        ohlcv_history (do run_index_step lưu trước đó) phải tạo ra bản
        ghi long_term_screener_report, giống hệt cách hoạt động với cổ
        phiếu thường."""
        storage = Storage(db_path=":memory:")
        collector = DataCollector(MockDataSource())
        run_index_step(collector, storage, "VN30", config={})

        run_long_term_screener_step(storage, {"VN30": "index"})

        record = storage.get_latest("long_term_screener_report", "VN30")
        assert record is not None
        assert record["data"]["sector"] == "index"
        assert "regime_fast" in record["data"]
        assert "regime_ensemble" in record["data"]

        # Bổ sung 07/09/2026: cùng lượt chạy còn lưu THÊM bộ lọc tổ hợp
        # theo năm, tái sử dụng regime_ensemble vừa tính (không fit lại
        # Markov lần 2).
        record_to_hop = storage.get_latest("chien_luoc_to_hop_theo_nam", "VN30")
        assert record_to_hop is not None
        assert record_to_hop["data"]["sector"] == "index"
        assert isinstance(record_to_hop["data"]["ket_qua"], list)
        storage.close()

    def test_ma_da_co_long_term_screener_report_tu_truoc_van_duoc_bo_sung_chien_luoc_to_hop(self):
        """Sự cố thực tế 08/09/2026: mục "Lọc bộ chỉ số/tổ hợp theo Năm &
        Giai đoạn" trên dashboard luôn báo "chưa có dữ liệu" dù đã chạy
        đầy đủ `run_full_market.py` — vì checkpoint CŨ dùng CHUNG 1 điều
        kiện `continue` cho cả 2 category: mã nào ĐÃ có
        `long_term_screener_report` TỪ TRƯỚC (VD tính từ đợt chạy trước
        khi tính năng tổ hợp-theo-năm ra đời) sẽ bị bỏ qua VĨNH VIỄN,
        không bao giờ có `chien_luoc_to_hop_theo_nam` dù chạy lại bao
        nhiêu lần. Mô phỏng đúng kịch bản: seed sẵn `long_term_screener_report`
        (giống mã đã tính từ trước), rồi gọi `run_long_term_screener_step()`
        BÌNH THƯỜNG (không `force_recompute`) -> `chien_luoc_to_hop_theo_nam`
        PHẢI được tính bổ sung, không bị bỏ qua theo mã kia."""
        storage = Storage(db_path=":memory:")
        collector = DataCollector(MockDataSource())
        run_index_step(collector, storage, "VN30", config={})

        # Giả lập mã ĐÃ được tính long_term_screener_report từ 1 đợt chạy
        # TRƯỚC (trước khi có category chien_luoc_to_hop_theo_nam).
        storage.save("long_term_screener_report", "VN30", {
            "sector": "index", "updated_at": "2026-01-01T00:00:00",
            "regime_fast": {"current": None, "best_strategy": None, "results": {}},
            "regime_ensemble": {"current": None, "best_strategy": None, "results": {}},
        })
        assert storage.get_latest("chien_luoc_to_hop_theo_nam", "VN30") is None

        run_long_term_screener_step(storage, {"VN30": "index"})

        record_to_hop = storage.get_latest("chien_luoc_to_hop_theo_nam", "VN30")
        assert record_to_hop is not None, (
            "chien_luoc_to_hop_theo_nam phải được tính bổ sung dù "
            "long_term_screener_report đã có sẵn từ trước"
        )
        assert isinstance(record_to_hop["data"]["ket_qua"], list)
        storage.close()
