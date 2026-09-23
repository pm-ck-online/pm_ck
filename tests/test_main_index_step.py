"""Test cho `main.run_index_step()` — bổ sung 27/08/2026 theo yêu cầu
người dùng thêm mã VN30 (và các chỉ số khác) vào mục "🎭 Tính cách giao
dịch từng mã" / "🧮 Cổ phiếu dài hạn" trên dashboard.

Cả 2 mục đó đọc dữ liệu GENERIC qua `storage.query_all_keys(...)`, không
lọc riêng theo `watchlist.symbols` — nên chỉ cần `run_index_step()` cũng
lưu category `stock_character` (giống cổ phiếu thường) là chỉ số sẽ TỰ
ĐỘNG xuất hiện trên dashboard, không cần sửa gì thêm ở `dashboard/app.py`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.data_collector import DataCollector, MockDataSource
from core.storage import Storage
from main import (
    SO_NGAY_LAM_MOI_CO_PHIEU_DAI_HAN,
    _da_qua_han_lam_moi_co_phieu_dai_han,
    _thieu_bo_chi_so_moi_trong_long_term_screener_report,
    run_index_step,
    run_long_term_screener_step,
    run_vn30f1m_step,
)


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


class TestRunVn30F1mStep:
    """Bổ sung 23/09/2026 (yêu cầu người dùng): tự động lấy dữ liệu HĐTL
    VN30F1M hàng ngày giống 1 mã bình thường — dùng `DataCollector.get_ohlcv()`
    (endpoint cổ phiếu thường), KHÁC `run_index_step()` (dùng
    `get_index_ohlcv()`, endpoint chỉ số riêng) — đã kiểm chứng thực tế
    vnstock TỰ nhận diện mã phái sinh qua đúng endpoint cổ phiếu thường."""

    def test_luu_ohlcv_va_indicator_snapshot_kem_atr14(self):
        from core.market_breadth import calculate_atr

        storage = Storage(db_path=":memory:")
        collector = DataCollector(MockDataSource())

        snapshot = run_vn30f1m_step(collector, storage, "VN30F1M", config={})

        assert snapshot is not None
        assert storage.get_latest("ohlcv_history", "VN30F1M") is not None
        record = storage.get_latest("indicator_snapshot", "VN30F1M")
        assert record is not None
        assert "atr14" in record["data"]  # có tính thêm ATR14 (dùng cho mục HĐTL VN30)

        # Đối chiếu chéo: ATR14 lưu lại phải KHỚP với gọi calculate_atr()
        # trực tiếp trên CÙNG dữ liệu (đã có test riêng cho calculate_atr()
        # — ở đây chỉ xác nhận run_vn30f1m_step() NỐI ĐÚNG kết quả đó vào
        # snapshot, không tính sai/tính thiếu).
        df_doi_chieu = collector.get_ohlcv("VN30F1M", timeframe="day")
        atr14_ky_vong = calculate_atr(df_doi_chieu, 14).iloc[-1]
        assert record["data"]["atr14"] == pytest.approx(float(atr14_ky_vong))

    def test_cung_tinh_ca_tinh_cach_giao_dich(self):
        """Giống chỉ số (run_index_step) — VN30F1M cũng phải xuất hiện ở
        mục "Tính cách giao dịch từng mã" (đọc storage.query_all_keys("stock_character"))."""
        storage = Storage(db_path=":memory:")
        collector = DataCollector(MockDataSource())

        run_vn30f1m_step(collector, storage, "VN30F1M", config={})

        character_record = storage.get_latest("stock_character", "VN30F1M")
        assert character_record is not None
        assert character_record["data"].get("nhan_tinh_cach") is not None
        storage.close()

    def test_loi_lay_du_lieu_khong_luu_gi_va_tra_ve_none(self):
        class _ThatBaiSource(MockDataSource):
            def fetch_ohlcv(self, symbol, timeframe="day"):
                raise RuntimeError("giả lập lỗi mạng")

        storage = Storage(db_path=":memory:")
        collector = DataCollector(_ThatBaiSource())

        ket_qua = run_vn30f1m_step(collector, storage, "VN30F1M", config={})

        assert ket_qua is None
        assert storage.get_latest("ohlcv_history", "VN30F1M") is None
        assert storage.get_latest("stock_character", "VN30F1M") is None
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
        # TRƯỚC (trước khi có category chien_luoc_to_hop_theo_nam) — dùng
        # updated_at MỚI (hôm nay) để KHÔNG bị coi là quá hạn làm mới (xem
        # TestLamMoiTheoHanCoPhieuDaiHan bên dưới cho riêng hành vi quá hạn).
        storage.save("long_term_screener_report", "VN30", {
            "sector": "index", "updated_at": datetime.now().isoformat(),
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


# ==============================================================================
# Test: _da_qua_han_lam_moi_co_phieu_dai_han — hàm thuần túy
# ==============================================================================

class TestDaQuaHanLamMoiCoPhieuDaiHan:
    def test_record_none_can_tinh_lai(self):
        assert _da_qua_han_lam_moi_co_phieu_dai_han(None, 7) is True

    def test_thieu_updated_at_can_tinh_lai(self):
        assert _da_qua_han_lam_moi_co_phieu_dai_han({"data": {}}, 7) is True

    def test_updated_at_khong_doc_duoc_can_tinh_lai(self):
        record = {"data": {"updated_at": "khong-phai-ngay-thang"}}
        assert _da_qua_han_lam_moi_co_phieu_dai_han(record, 7) is True

    def test_moi_hom_nay_chua_can_tinh_lai(self):
        record = {"data": {"updated_at": datetime.now().isoformat()}}
        assert _da_qua_han_lam_moi_co_phieu_dai_han(record, 7) is False

    def test_5_ngay_truoc_chua_qua_han_7_ngay(self):
        record = {"data": {"updated_at": (datetime.now() - timedelta(days=5)).isoformat()}}
        assert _da_qua_han_lam_moi_co_phieu_dai_han(record, 7) is False

    def test_10_ngay_truoc_da_qua_han_7_ngay(self):
        record = {"data": {"updated_at": (datetime.now() - timedelta(days=10)).isoformat()}}
        assert _da_qua_han_lam_moi_co_phieu_dai_han(record, 7) is True

    def test_dieu_kien_la_lon_hon_khong_phai_lon_hon_bang(self):
        # Dùng 6.99/7.01 ngày (thay vì đúng 7.0) để tránh sai số nhỏ giữa
        # lúc tạo timestamp và lúc so sánh làm test không ổn định.
        record_chua_qua = {"data": {"updated_at": (datetime.now() - timedelta(days=6.99)).isoformat()}}
        record_da_qua = {"data": {"updated_at": (datetime.now() - timedelta(days=7.01)).isoformat()}}
        assert _da_qua_han_lam_moi_co_phieu_dai_han(record_chua_qua, 7) is False
        assert _da_qua_han_lam_moi_co_phieu_dai_han(record_da_qua, 7) is True


# ==============================================================================
# Test: _thieu_bo_chi_so_moi_trong_long_term_screener_report — sự cố thực
# tế 24/09/2026 (LẦN THỨ 4 cùng dạng lỗi — xem 4n/4o trong CLAUDE.md): sau
# khi thêm bộ chỉ số MỚI "Cuối tháng + RSI70" vào xay_8_bo_chi_so(), dữ
# liệu ĐÃ TÍNH TRƯỚC ĐÓ (dù còn mới theo NGÀY, trong hạn 7 ngày) vẫn
# THIẾU bộ chỉ số này — checkpoint theo hạn ngày (4o) không phát hiện
# được, cần thêm kiểm tra CẤU TRÚC riêng.
# ==============================================================================

class TestThieuBoChiSoMoiTrongLongTermScreenerReport:
    def _ket_qua_mau(self, ten_bo_chi_so_list):
        return {ten: {"n_trades": 0} for ten in ten_bo_chi_so_list}

    def test_record_none_khong_can_tinh_lai_o_day(self):
        # None đã được _da_qua_han_lam_moi_co_phieu_dai_han() xử lý riêng
        # -> hàm này PHẢI trả về False để tránh trùng logic/gây nhầm lẫn.
        assert _thieu_bo_chi_so_moi_trong_long_term_screener_report(None) is False

    def test_du_ca_9_bo_khong_can_tinh_lai(self):
        from core.long_term_indicator_backtest import TEN_CAC_BO_CHI_SO_DON_LE
        record = {"data": {
            "regime_fast": {"results": self._ket_qua_mau(TEN_CAC_BO_CHI_SO_DON_LE)},
            "regime_ensemble": {"results": self._ket_qua_mau(TEN_CAC_BO_CHI_SO_DON_LE)},
        }}
        assert _thieu_bo_chi_so_moi_trong_long_term_screener_report(record) is False

    def test_thieu_cuoi_thang_rsi70_can_tinh_lai(self):
        # Mô phỏng ĐÚNG dữ liệu cũ (tính trước 24/09/2026, chỉ có 8 bộ).
        ten_8_bo_cu = [
            "MA20 (Giá cắt MA20)", "EMA50/EMA200 (Golden/Death Cross)",
            "RSI14 (Quá mua/Quá bán 30-70)", "Bollinger Breakout + Volume",
            "Bollinger Bounce (mua đáy dải dưới)", "Volume Breakout + MA20",
            "Kết hợp: Trend Filter EMA + RSI", "Mua và giữ (Buy & Hold)",
        ]
        record = {"data": {
            "regime_fast": {"results": self._ket_qua_mau(ten_8_bo_cu)},
            "regime_ensemble": {"results": self._ket_qua_mau(ten_8_bo_cu)},
        }}
        assert _thieu_bo_chi_so_moi_trong_long_term_screener_report(record) is True

    def test_thieu_ca_regime_fast_lan_regime_ensemble_van_can_tinh_lai(self):
        record = {"data": {}}
        assert _thieu_bo_chi_so_moi_trong_long_term_screener_report(record) is True


# ==============================================================================
# Test: run_long_term_screener_step tự làm mới theo hạn — sự cố thực tế
# 15/09/2026: dù chạy batch hàng ngày, "Cổ phiếu dài hạn" của SSI vẫn báo
# "Cập nhật lần cuối: 26/08/2026" vì checkpoint cũ chỉ hỏi "đã có chưa",
# không hỏi "đã CŨ chưa". Đã sửa: tự tính lại nếu quá
# SO_NGAY_LAM_MOI_CO_PHIEU_DAI_HAN (7) ngày, không cần force_recompute.
# ==============================================================================

class TestLamMoiTheoHanCoPhieuDaiHan:
    def _seed_bao_cao_cu(self, storage: Storage, updated_at: str) -> None:
        # `results` liệt kê ĐỦ TẤT CẢ bộ chỉ số HIỆN TẠI (không rỗng) —
        # mô phỏng dữ liệu THỰC SỰ đầy đủ/mới, khác với kịch bản "thiếu
        # bộ chỉ số mới" (test riêng bên dưới, dùng dữ liệu CHỈ có 8 bộ cũ).
        from core.long_term_indicator_backtest import TEN_CAC_BO_CHI_SO_DON_LE
        ket_qua_mau = {ten: {"n_trades": 0} for ten in TEN_CAC_BO_CHI_SO_DON_LE}
        storage.save("long_term_screener_report", "VN30", {
            "sector": "index", "updated_at": updated_at,
            "regime_fast": {"current": None, "best_strategy": None, "results": ket_qua_mau},
            "regime_ensemble": {"current": None, "best_strategy": None, "results": ket_qua_mau},
        })
        storage.save("chien_luoc_to_hop_theo_nam", "VN30", {
            "sector": "index", "updated_at": updated_at, "ket_qua": [],
        })

    def _seed_bao_cao_thieu_bo_chi_so_moi(self, storage: Storage, updated_at: str) -> None:
        """Mô phỏng dữ liệu tính TRƯỚC 24/09/2026 — chỉ có 8 bộ cũ, THIẾU
        "Cuối tháng + RSI70" — dù `updated_at` còn MỚI (trong hạn 7 ngày)."""
        ten_8_bo_cu = [
            "MA20 (Giá cắt MA20)", "EMA50/EMA200 (Golden/Death Cross)",
            "RSI14 (Quá mua/Quá bán 30-70)", "Bollinger Breakout + Volume",
            "Bollinger Bounce (mua đáy dải dưới)", "Volume Breakout + MA20",
            "Kết hợp: Trend Filter EMA + RSI", "Mua và giữ (Buy & Hold)",
        ]
        ket_qua_mau = {ten: {"n_trades": 0} for ten in ten_8_bo_cu}
        storage.save("long_term_screener_report", "VN30", {
            "sector": "index", "updated_at": updated_at,
            "regime_fast": {"current": None, "best_strategy": None, "results": ket_qua_mau},
            "regime_ensemble": {"current": None, "best_strategy": None, "results": ket_qua_mau},
        })
        storage.save("chien_luoc_to_hop_theo_nam", "VN30", {
            "sector": "index", "updated_at": updated_at, "ket_qua": [],
        })

    def test_du_lieu_qua_han_tu_dong_duoc_tinh_lai(self):
        storage = Storage(db_path=":memory:")
        collector = DataCollector(MockDataSource())
        run_index_step(collector, storage, "VN30", config={})

        ngay_cu = (datetime.now() - timedelta(days=SO_NGAY_LAM_MOI_CO_PHIEU_DAI_HAN + 3)).isoformat()
        self._seed_bao_cao_cu(storage, ngay_cu)

        run_long_term_screener_step(storage, {"VN30": "index"})

        record_report = storage.get_latest("long_term_screener_report", "VN30")
        record_to_hop = storage.get_latest("chien_luoc_to_hop_theo_nam", "VN30")
        assert record_report["data"]["updated_at"] != ngay_cu, (
            "Bản ghi quá hạn phải được tính lại với updated_at MỚI"
        )
        assert record_to_hop["data"]["updated_at"] != ngay_cu
        assert (datetime.now() - datetime.fromisoformat(record_report["data"]["updated_at"])) < timedelta(minutes=5)

    def test_du_lieu_con_moi_khong_bi_tinh_lai(self):
        storage = Storage(db_path=":memory:")
        collector = DataCollector(MockDataSource())
        run_index_step(collector, storage, "VN30", config={})

        ngay_moi = (datetime.now() - timedelta(days=2)).isoformat()
        self._seed_bao_cao_cu(storage, ngay_moi)

        run_long_term_screener_step(storage, {"VN30": "index"})

        record_report = storage.get_latest("long_term_screener_report", "VN30")
        record_to_hop = storage.get_latest("chien_luoc_to_hop_theo_nam", "VN30")
        # Chưa quá hạn -> KHÔNG bị tính lại -> updated_at giữ nguyên giá
        # trị đã seed (không có bản ghi MỚI nào được lưu đè lên).
        assert record_report["data"]["updated_at"] == ngay_moi
        assert record_to_hop["data"]["updated_at"] == ngay_moi
        storage.close()

    def test_du_moi_nhung_thieu_bo_chi_so_moi_van_duoc_tinh_lai(self):
        """Sự cố thực tế 24/09/2026 (LẦN THỨ 4 cùng dạng lỗi — xem 4n/4o
        CLAUDE.md): dữ liệu CÒN MỚI theo ngày (2 ngày trước, trong hạn 7
        ngày) nhưng THIẾU bộ chỉ số "Cuối tháng + RSI70" (tính bằng code
        cũ trước khi thêm bộ này) — PHẢI vẫn được tính lại, không được để
        "mắc kẹt" thiếu vĩnh viễn cho tới lần quá hạn tiếp theo."""
        storage = Storage(db_path=":memory:")
        collector = DataCollector(MockDataSource())
        run_index_step(collector, storage, "VN30", config={})

        ngay_moi = (datetime.now() - timedelta(days=2)).isoformat()
        self._seed_bao_cao_thieu_bo_chi_so_moi(storage, ngay_moi)

        run_long_term_screener_step(storage, {"VN30": "index"})

        record_report = storage.get_latest("long_term_screener_report", "VN30")
        record_to_hop = storage.get_latest("chien_luoc_to_hop_theo_nam", "VN30")
        assert record_report["data"]["updated_at"] != ngay_moi, (
            "Dữ liệu thiếu bộ chỉ số mới phải được tính lại dù còn MỚI theo ngày"
        )
        assert record_to_hop["data"]["updated_at"] != ngay_moi
        ket_qua_fast = record_report["data"]["regime_fast"]["results"]
        assert "Cuối tháng + RSI70" in ket_qua_fast
        storage.close()
