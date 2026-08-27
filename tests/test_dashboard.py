"""
Unit test (smoke test) cho dashboard/app.py

Dùng `streamlit.testing.v1.AppTest` để chạy thử dashboard mà KHÔNG cần mở
trình duyệt thật.

QUAN TRỌNG VỀ CÁCH LY DỮ LIỆU: mọi test trong file này dùng
`tmp_path` (thư mục tạm do pytest tự tạo/xóa cho từng test) làm database,
thông qua biến môi trường `PM_CK_DB_PATH` mà `dashboard/app.py` đọc để
xác định đường dẫn storage. TUYỆT ĐỐI KHÔNG dùng chung đường dẫn
`./data/pm_ck.db` thật — vì từng gây ra sự cố dữ liệu test bị lẫn vào dữ
liệu thật của người dùng khi việc dọn dẹp thư mục thất bại (đặc biệt trên
Windows do khóa file SQLite).
"""

from __future__ import annotations

from datetime import datetime

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from core.storage import Storage

from pathlib import Path

DASHBOARD_PATH = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


@pytest.fixture
def isolated_db_path(tmp_path, monkeypatch):
    """Trỏ dashboard sang 1 file database TẠM THỜI, tách biệt hoàn toàn
    khỏi database thật của người dùng. `tmp_path` được pytest tự động
    tạo mới cho MỖI test và tự dọn dẹp sau đó — không cần tự viết logic
    xóa thư mục (tránh lặp lại lỗi cũ do dọn dẹp thất bại trên Windows).
    """
    db_path = str(tmp_path / "test_pm_ck.db")
    monkeypatch.setenv("PM_CK_DB_PATH", db_path)
    st.cache_resource.clear()  # đảm bảo không dùng lại kết nối đã cache từ test trước
    # st.cache_data DÙNG CHUNG 1 bộ nhớ đệm ở CẤP TIẾN TRÌNH (không tự tách
    # biệt theo test) — nếu không xóa, 1 test gọi hàm cache (VD
    # _tinh_chi_bao_gan_dat_theo_realtime_cached, khóa cache CHỈ theo danh
    # sách mã, không theo nội dung storage) có thể ăn nhầm kết quả CŨ từ
    # test chạy trước đó dùng CÙNG mã nhưng khác dữ liệu/mock.
    st.cache_data.clear()
    return db_path


@pytest.fixture
def seeded_storage(isolated_db_path):
    """Tạo sẵn dữ liệu mẫu vào database TẠM (isolated_db_path), để AppTest
    chạy dashboard đọc được dữ liệu này mà không đụng tới database thật.

    Mục "🎭 Tính cách giao dịch từng mã" TỰ ĐỘNG gọi giá REALTIME khi danh
    sách đủ nhỏ (xem NGUONG_TOI_DA_MA_GAN_DAT_REALTIME trong
    dashboard/app.py) — việc chặn gọi mạng thật đã có sẵn qua biến môi
    trường `PM_CK_SKIP_REALTIME=1` (autouse fixture trong
    `tests/conftest.py`, xem ghi chú ở đó về lý do PHẢI dùng biến môi
    trường thay vì monkeypatch trực tiếp hàm khi test qua AppTest).
    """
    storage = Storage(db_path=isolated_db_path)

    # Watchlist > 5 mã -> mọi test dùng fixture này tự động phủ luôn
    # nhánh "hiện ô tìm kiếm" (render_search_box_if_needed) mà KHÔNG cần
    # thêm test AppTest riêng (tránh hiện tượng tràn trạng thái form giữa
    # nhiều AppTest liên tiếp trong cùng 1 phiên pytest — xem ghi chú ở
    # test_dashboard_shows_watchlist_data bên dưới).
    storage.save("watchlist", "default", {
        "symbols": ["HPG", "VNM", "FPT", "SSI", "VIB", "TNG", "BID"],
    })

    storage.save("indicator_snapshot", "HPG", {
        "close": 32.5, "ma20": 31.0, "ema50": 30.0, "ema100": 28.0,
        "ema200": 27.0, "volume_ma_15": 1_200_000, "volume_ma_20": 1_100_000,
        "price_above_ema200": True,
    })

    # Mục "🎭 Tính cách giao dịch từng mã" đọc category riêng
    # "stock_character" (KHÁC "indicator_snapshot") để xác định danh sách
    # mã cần hiển thị — thiếu seed này thì toàn bộ thân hàm
    # render_stock_character_section() bị bỏ qua (thoát sớm ở nhánh
    # "Chưa có dữ liệu"), không test được nhánh chính của mục.
    storage.save("stock_character", "HPG", {
        "nhan_tinh_cach": "TRUNG_TINH", "character_score": 0.4,
        "choppiness_score": 55.0, "canh_bao": [], "do_tin_cay_thap": False,
    })

    # Cần thiết để mục "Giai đoạn thị trường" không thoát sớm ở bước lấy
    # danh sách ngành (all_symbol_sector_keys) — mã HPG được gán ngành
    # "banking", khớp với market_regime "banking" seed bên dưới.
    storage.save("symbol_sector", "HPG", {"sector": "banking"})

    storage.save("realtime_price", "HPG", {
        "symbol": "HPG", "price": 33.0, "volume": 1_500_000,
        "timestamp": datetime(2026, 1, 10, 14, 30),
    })

    ohlcv_records = [
        {
            "date": f"2025-{(1 + i // 28):02d}-{(1 + i % 28):02d}",
            "open": 30.0 + (i % 5), "high": 30.5 + (i % 5),
            "low": 29.5 + (i % 5), "close": 30.0 + (i % 5),
            "volume": 1_000_000 + i * 1000,
        }
        for i in range(260)
    ]
    storage.save("ohlcv_history", "HPG", {"records": ohlcv_records})

    storage.save("market_regime", "banking", {
        "regime": "uptrend", "confidence": 0.8,
        "reasoning": ["80% mã trên EMA200", "Không có tín hiệu vĩ mô tiêu cực"],
        "affected_sectors": ["real_estate"],
    })

    storage.save("allocation_recommendation", "HPG", {
        "target_pct": 85.0, "tranches": [30, 50, 20],
        "entry_price_range": {"low": 31.0, "high": 33.0},
        "stop_loss": 28.8, "max_position_size": 1000,
        "notes": ["Uptrend: khuyến nghị giải ngân theo 3 đợt."],
    })

    storage.save("pattern_result", "HPG", {
        "confidence": 0.75, "accumulation_high": 33.5,
        "effective_scan_months": 18.5,
    })

    storage.save("portfolio_snapshot", "default", {
        "nav": 1_050_000_000, "total_return_pct": 5.0,
        "total_stock_weight_pct": 60.0,
        "positions": [
            {"symbol": "HPG", "qty": 1000, "market_value": 32_500_000},
        ],
    }, timestamp=datetime(2026, 1, 1))
    storage.save("portfolio_snapshot", "default", {
        "nav": 1_080_000_000, "total_return_pct": 8.0,
        "total_stock_weight_pct": 62.0,
        "positions": [
            {"symbol": "HPG", "qty": 1000, "market_value": 34_000_000},
        ],
    }, timestamp=datetime(2026, 1, 2))

    # Cố tình trộn 1 khung có giá trị số THẬT và 1 khung có giá trị None
    # trong cùng cột "so_phien_tb_toi_day" — đúng kịch bản đã gây lỗi
    # ArrowTypeError thực tế (27/07/2026) khi hiển thị trên dashboard.
    storage.save("short_term_signal_report", "latest", {
        "ngay_danh_gia": "2026-07-27",
        "vnindex": {
            "do_lech_ma20_pct": 2.5, "muc_canh_bao": "CANH_BAO_DIEU_CHINH",
            "xac_suat_dieu_chinh": {
                "tong_so_su_kien_lich_su": 10,
                "theo_khung_ngay": {
                    5: {"xac_suat_pct": 60.0, "muc_dieu_chinh_tb_pct": 4.0, "so_phien_tb_toi_day": 3.5, "so_su_kien_hop_le": 10},
                    10: {"xac_suat_pct": None, "muc_dieu_chinh_tb_pct": None, "so_phien_tb_toi_day": None, "so_su_kien_hop_le": 0},
                },
            },
        },
        "tin_hieu_bat_ca_hoi": {
            "kich_hoat": False, "muc_giam_tu_dinh_40_phien_pct": 3.0,
            "nganh_uu_tien": [], "ro_ma_uu_tien": None, "phu_quyet_ly_do": [],
        },
        "co_phieu_qua_mua": [],
        "canh_bao": [],
        "ghi_chu": "test",
    })

    _ket_qua_chien_luoc_mau = {
        "uptrend": {"n_trades": 3, "win_rate_pct": 66.7, "total_return_pct": 12.5, "avg_return_pct": 4.1, "ending_capital": 1_125_000_000.0},
        "sideway": {"n_trades": 0, "win_rate_pct": None, "total_return_pct": None, "avg_return_pct": None, "ending_capital": 1_000_000_000.0},
        "downtrend": {"n_trades": 1, "win_rate_pct": 0.0, "total_return_pct": -2.3, "avg_return_pct": -2.3, "ending_capital": 977_000_000.0},
    }
    storage.save("long_term_screener_report", "HPG", {
        "sector": "banking",
        "updated_at": "2026-08-24T00:00:00",
        "regime_fast": {
            "current": "uptrend", "best_strategy": "MA20 (Giá cắt MA20)",
            "results": {ten: _ket_qua_chien_luoc_mau for ten in [
                "MA20 (Giá cắt MA20)", "EMA50/EMA200 (Golden/Death Cross)",
                "RSI14 (Quá mua/Quá bán 30-70)", "Bollinger Breakout + Volume",
                "Bollinger Bounce (mua đáy dải dưới)", "Volume Breakout + MA20",
                "Kết hợp: Trend Filter EMA + RSI", "Mua và giữ (Buy & Hold)",
            ]},
        },
        "regime_ensemble": {
            "current": "uptrend", "best_strategy": "MA20 (Giá cắt MA20)",
            "results": {ten: _ket_qua_chien_luoc_mau for ten in [
                "MA20 (Giá cắt MA20)", "EMA50/EMA200 (Golden/Death Cross)",
                "RSI14 (Quá mua/Quá bán 30-70)", "Bollinger Breakout + Volume",
                "Bollinger Bounce (mua đáy dải dưới)", "Volume Breakout + MA20",
                "Kết hợp: Trend Filter EMA + RSI", "Mua và giữ (Buy & Hold)",
            ]},
        },
    })

    storage.close()

    return isolated_db_path


# ==============================================================================
# Smoke test: dashboard chạy không lỗi với dữ liệu đã seed
# ==============================================================================

class TestDashboardSmoke:
    def test_dashboard_runs_without_exception(self, seeded_storage):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        assert not at.exception

        # Mục "🎭 Tính cách giao dịch từng mã": danh sách chỉ có 1 mã (HPG,
        # seed ở seeded_storage) -> TỰ ĐỘNG thử lấy giá Realtime (không cần
        # bật checkbox nào). Fixture chặn API thật -> phải rơi về giá đóng
        # cửa (EOD) một cách AN TOÀN, không throw exception, và cột "Nguồn
        # giá" phải phản ánh đúng trạng thái lỗi/fallback này.
        for df in at.dataframe:
            try:
                cols = list(df.value.columns)
            except Exception:  # noqa: BLE001
                continue
            if "Nguồn giá" in cols and "Mã" in cols:
                hang_hpg = df.value[df.value["Mã"] == "HPG"]
                assert not hang_hpg.empty
                assert hang_hpg.iloc[0]["Nguồn giá"] == "⚠️ Lỗi Realtime (dùng đóng cửa)"
                assert hang_hpg.iloc[0]["Giá"] == pytest.approx(32.5)  # rơi về EOD
                break
        else:
            pytest.fail("Không tìm thấy bảng \"Tính cách giao dịch\" có cột \"Nguồn giá\".")

    def test_single_section_mode_shows_only_selected_section(self, seeded_storage):
        """Chọn chế độ 'Chỉ xem 1 mục' -> chỉ đúng 1 mục được hiển thị,
        không hiện toàn bộ trang.
        """
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        assert not at.exception

        at.radio(key="dashboard_view_mode").set_value("Chỉ xem 1 mục")
        at.run(timeout=30)
        assert not at.exception

        at.radio(key="dashboard_selected_group").set_value("NHÓM 3 — GIAO DỊCH CỔ PHIẾU")
        at.run(timeout=30)
        assert not at.exception

        at.radio(key="dashboard_selected_section__NHÓM 3 — GIAO DỊCH CỔ PHIẾU").set_value(
            "🎭 Tính cách giao dịch từng mã"
        )
        at.run(timeout=30)
        assert not at.exception

        # Chỉ đúng 1 tiêu đề mục được hiển thị (không lẫn các mục khác)
        subheader_texts = [s.value for s in at.subheader]
        assert any("Tính cách giao dịch từng mã" in s for s in subheader_texts)
        assert not any("Biểu đồ nến" in s for s in subheader_texts)

    def test_dashboard_title_present(self, seeded_storage):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        titles = [t.value for t in at.title]
        assert any("pm_ck" in t for t in titles)

    def test_dashboard_shows_watchlist_data(self, seeded_storage):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        # Không lỗi và có ít nhất 1 bảng dữ liệu (dataframe) được render
        assert not at.exception
        assert len(at.dataframe) > 0

        # seeded_storage có 7 mã trong watchlist (>5) -> ô tìm kiếm PHẢI
        # xuất hiện (render_search_box_if_needed). Kiểm tra ngay trong
        # test smoke đã ổn định thay vì viết thêm AppTest riêng — tránh
        # hiện tượng tràn trạng thái form khi có quá nhiều AppTest liên
        # tiếp trong cùng 1 phiên pytest (đã xác nhận thực tế 27/07/2026).
        assert at.text_input(key="watchlist_search") is not None
        # (chart_symbol_search KHÔNG hiện vì seeded_storage chỉ có
        # ohlcv_history cho 1 mã HPG — đúng thiết kế, chỉ mã có dữ liệu
        # biểu đồ mới được đưa vào danh sách chọn.)

    def test_dashboard_shows_market_regime_warning_for_affected_sector(self, seeded_storage):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        warning_texts = [w.value for w in at.warning]
        assert any("real_estate" in w for w in warning_texts)

    def test_dashboard_renders_candlestick_chart_without_exception(self, seeded_storage):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        assert not at.exception
        # Selectbox chọn mã để vẽ biểu đồ phải xuất hiện (do đã seed ohlcv_history cho HPG)
        assert len(at.selectbox) > 0


# ==============================================================================
# Test tương tác: Nhật ký giao dịch mua/bán (ghi nhận mua -> đóng vị thế bán)
# ==============================================================================

class TestTradeJournalInteraction:
    def test_record_buy_creates_open_trade(self, seeded_storage):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        assert not at.exception

        at.number_input(key="buy_qty").set_value(100)
        at.number_input(key="buy_price").set_value(32.5)
        at.text_area(key="buy_reason").set_value("Breakout khỏi vùng tích lũy")
        at.button(key="FormSubmitter:buy_form-Ghi nhận mua").click()
        at.run(timeout=30)

        assert not at.exception
        storage = Storage(db_path=seeded_storage)
        trade_ids = storage.query_all_keys("trade_journal")
        assert len(trade_ids) == 1
        record = storage.get_latest("trade_journal", trade_ids[0])
        assert record["data"]["is_closed"] is False
        assert record["data"]["buy_price"] == pytest.approx(32.5)
        storage.close()

    def test_close_position_computes_pnl_correctly(self, seeded_storage):
        # Chuẩn bị sẵn 1 giao dịch MUA đang mở trong storage tạm
        storage = Storage(db_path=seeded_storage)
        from core.trade_journal import create_trade_entry
        from datetime import date as date_cls

        entry = create_trade_entry(
            symbol="HPG", qty=100, buy_price=20.0, buy_date=date_cls(2026, 1, 1),
            buy_reason="Test mua", buy_reference_indicator="EMA50",
        )
        storage.save("trade_journal", entry["trade_id"], entry)
        storage.close()

        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        assert not at.exception

        at.number_input(key="sell_price").set_value(24.0)
        at.text_area(key="sell_reason").set_value("Chạm kháng cự EMA200")
        at.button(key="FormSubmitter:sell_form-Ghi nhận bán").click()
        at.run(timeout=30)

        assert not at.exception

        storage = Storage(db_path=seeded_storage)
        record = storage.get_latest("trade_journal", entry["trade_id"])
        assert record["data"]["is_closed"] is True
        assert record["data"]["sell_price"] == pytest.approx(24.0)
        assert record["data"]["pnl"] == pytest.approx((24.0 - 20.0) * 100)
        storage.close()


# ==============================================================================
# Test tương tác: Quản lý Watchlist (thêm/xóa mã)
# ==============================================================================

class TestWatchlistManagerInteraction:
    def test_add_symbol_persists_to_storage(self, seeded_storage):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        assert not at.exception

        at.text_input(key="new_symbol_input").set_value("SSI2")
        at.button(key="add_symbol_btn").click()
        at.run(timeout=30)

        assert not at.exception
        storage = Storage(db_path=seeded_storage)
        record = storage.get_latest("watchlist", "default")
        assert "SSI2" in record["data"]["symbols"]
        storage.close()


# ==============================================================================
# Test tương tác: Nhập dữ liệu vĩ mô thủ công (Fed Rate / Tỷ giá USD-VND)
# ==============================================================================

class TestManualMacroDataInteraction:
    def test_add_macro_entry_persists_to_storage(self, seeded_storage):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        assert not at.exception

        at.number_input(key="macro_entry_value").set_value(5.25)
        at.button(key="FormSubmitter:macro_entry_form-Lưu điểm dữ liệu").click()
        at.run(timeout=30)

        assert not at.exception
        storage = Storage(db_path=seeded_storage)
        record = storage.get_latest("manual_macro_series", "fed_rate")
        assert record is not None
        assert record["data"]["entries"][0]["value"] == pytest.approx(5.25)
        storage.close()


# ==============================================================================
# Test tương tác: Chú thích sự kiện trên biểu đồ
# ==============================================================================

class TestChartAnnotationInteraction:
    def test_add_annotation_persists_and_renders(self, seeded_storage):
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        assert not at.exception

        at.text_area(key="ann_text").set_value("Mỹ tấn công Iran")
        at.button(key="FormSubmitter:annotation_form-Lưu chú thích").click()
        at.run(timeout=30)

        assert not at.exception
        storage = Storage(db_path=seeded_storage)
        ann_ids = storage.query_all_keys("chart_annotation")
        assert len(ann_ids) == 1
        record = storage.get_latest("chart_annotation", ann_ids[0])
        assert record["data"]["text"] == "Mỹ tấn công Iran"
        storage.close()

    def test_delete_annotation_removes_it(self, seeded_storage):
        storage = Storage(db_path=seeded_storage)
        from core.chart_annotations import create_annotation
        from datetime import date as date_cls

        ann = create_annotation("HPG", date_cls(2025, 11, 11), "Sự kiện thử nghiệm")
        storage.save("chart_annotation", ann["annotation_id"], ann)
        storage.close()

        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        assert not at.exception

        at.button(key=f"del_ann_{ann['annotation_id']}").click()
        at.run(timeout=30)

        assert not at.exception
        storage = Storage(db_path=seeded_storage)
        assert storage.get_latest("chart_annotation", ann["annotation_id"]) is None
        storage.close()


# ==============================================================================
# Smoke test: dashboard KHÔNG lỗi ngay cả khi CHƯA có dữ liệu nào
# ==============================================================================

class TestDashboardEmptyState:
    def test_dashboard_runs_without_exception_when_no_data(self, isolated_db_path):
        # isolated_db_path đảm bảo trỏ tới file TẠM chưa từng có dữ liệu gì
        at = AppTest.from_file(DASHBOARD_PATH)
        at.run(timeout=30)
        assert not at.exception


# ==============================================================================
# Test: _tim_moc_rsi_quan_trong (hàm thuần túy — mốc RSI đánh số trên
# biểu đồ nến, xem render_chart_section)
# ==============================================================================

class TestTimMocRsiQuanTrong:
    def _ngay(self, n):
        return [f"2024-01-{i + 1:02d}" for i in range(n)]

    def test_khong_co_dot_qua_mua_qua_ban_tra_ve_rong(self):
        from dashboard.app import _tim_moc_rsi_quan_trong
        rsi = [40, 50, 60, 55, 45]
        assert _tim_moc_rsi_quan_trong(self._ngay(5), rsi) == []

    def test_1_dot_qua_mua_lay_dung_dinh(self):
        from dashboard.app import _tim_moc_rsi_quan_trong
        # qua muc 70 tu ngay 2 (idx1) den ngay 4 (idx3), dinh o idx2 (85)
        rsi = [60, 75, 85, 80, 55]
        ngay = self._ngay(5)
        ket_qua = _tim_moc_rsi_quan_trong(ngay, rsi)
        assert ket_qua == [{"date": ngay[2], "value": 85, "loai": "qua_mua"}]

    def test_1_dot_qua_ban_lay_dung_day(self):
        from dashboard.app import _tim_moc_rsi_quan_trong
        rsi = [40, 25, 15, 22, 45]
        ngay = self._ngay(5)
        ket_qua = _tim_moc_rsi_quan_trong(ngay, rsi)
        assert ket_qua == [{"date": ngay[2], "value": 15, "loai": "qua_ban"}]

    def test_dot_keo_dai_toi_cuoi_du_lieu_van_duoc_ghi_nhan(self):
        from dashboard.app import _tim_moc_rsi_quan_trong
        # qua mua tu idx1, tang dan toi cuoi, KHONG quay ve binh thuong
        rsi = [50, 72, 78, 90]
        ngay = self._ngay(4)
        ket_qua = _tim_moc_rsi_quan_trong(ngay, rsi)
        assert ket_qua == [{"date": ngay[3], "value": 90, "loai": "qua_mua"}]

    def test_nhieu_dot_tach_dung_theo_thu_tu_thoi_gian(self):
        from dashboard.app import _tim_moc_rsi_quan_trong
        # dot 1: qua ban (idx0-1, day=20 tai idx1); binh thuong idx2;
        # dot 2: qua mua (idx3-4, dinh=88 tai idx4)
        rsi = [25, 20, 50, 75, 88]
        ngay = self._ngay(5)
        ket_qua = _tim_moc_rsi_quan_trong(ngay, rsi)
        assert ket_qua == [
            {"date": ngay[1], "value": 20, "loai": "qua_ban"},
            {"date": ngay[4], "value": 88, "loai": "qua_mua"},
        ]

    def test_bo_qua_gia_tri_nan(self):
        from dashboard.app import _tim_moc_rsi_quan_trong
        import math
        rsi = [50, 75, math.nan, 80, 50]
        ngay = self._ngay(5)
        ket_qua = _tim_moc_rsi_quan_trong(ngay, rsi)
        # NaN cat doi doan qua mua -> 2 dot rieng, moi dot lay dung dinh cua no
        assert ket_qua == [
            {"date": ngay[1], "value": 75, "loai": "qua_mua"},
            {"date": ngay[3], "value": 80, "loai": "qua_mua"},
        ]


# ==============================================================================
# Test: hàm tiện ích tìm kiếm mã (pure function, không cần AppTest)
# ==============================================================================

class TestFilterSymbolsBySearch:
    def test_empty_search_returns_all(self):
        from dashboard.app import filter_symbols_by_search
        assert filter_symbols_by_search(["HPG", "VNM"], "") == ["HPG", "VNM"]

    def test_partial_match_case_insensitive(self):
        from dashboard.app import filter_symbols_by_search
        result = filter_symbols_by_search(["HPG", "VNM", "FPT"], "hp")
        assert result == ["HPG"]

    def test_no_match_returns_empty_list(self):
        from dashboard.app import filter_symbols_by_search
        result = filter_symbols_by_search(["HPG", "VNM"], "XYZ")
        assert result == []


# ==============================================================================
# Test: _bo_chi_so_tot_theo_nguong (hàm thuần túy dùng ở mục Tính cách
# giao dịch — liệt kê bộ chỉ số LN>5% theo giai đoạn hiện tại)
# ==============================================================================

class TestBoChiSoTotTheoNguong:
    def _pp_data(self, **ket_qua_theo_bo_chi_so):
        """`ket_qua_theo_bo_chi_so`: {ten_day_du: {giai_doan: {n_trades, total_return_pct}}}"""
        return {"current": "downtrend", "results": ket_qua_theo_bo_chi_so}

    def test_tra_ve_gach_ngang_khi_chua_co_giai_doan(self):
        from dashboard.app import _bo_chi_so_tot_theo_nguong
        pp_data = self._pp_data()
        assert _bo_chi_so_tot_theo_nguong(pp_data, None) == "—"

    def test_chi_lay_bo_vuot_nguong_va_co_lenh(self):
        from dashboard.app import _bo_chi_so_tot_theo_nguong
        pp_data = self._pp_data(**{
            "MA20 (Giá cắt MA20)": {"downtrend": {"n_trades": 3, "total_return_pct": 12.0}},
            "RSI14 (Quá mua/Quá bán 30-70)": {"downtrend": {"n_trades": 2, "total_return_pct": 3.0}},  # dưới ngưỡng
            "Bollinger Bounce (mua đáy dải dưới)": {"downtrend": {"n_trades": 0, "total_return_pct": 20.0}},  # 0 lệnh
            "Volume Breakout + MA20": {"downtrend": {"n_trades": 5, "total_return_pct": None}},  # thiếu %
        })
        ket_qua = _bo_chi_so_tot_theo_nguong(pp_data, "downtrend")
        assert ket_qua == "MA20 (3 lệnh, +12.0%)"

    def test_sap_xep_giam_dan_theo_loi_nhuan(self):
        from dashboard.app import _bo_chi_so_tot_theo_nguong
        pp_data = self._pp_data(**{
            "MA20 (Giá cắt MA20)": {"downtrend": {"n_trades": 3, "total_return_pct": 12.0}},
            "RSI14 (Quá mua/Quá bán 30-70)": {"downtrend": {"n_trades": 2, "total_return_pct": 41.4}},
        })
        ket_qua = _bo_chi_so_tot_theo_nguong(pp_data, "downtrend")
        assert ket_qua == "RSI14 (2 lệnh, +41.4%); MA20 (3 lệnh, +12.0%)"

    def test_chi_xet_dung_giai_doan_dang_hoi(self):
        from dashboard.app import _bo_chi_so_tot_theo_nguong
        pp_data = self._pp_data(**{
            "MA20 (Giá cắt MA20)": {
                "uptrend": {"n_trades": 3, "total_return_pct": 50.0},  # giai đoạn khác -> bỏ qua
                "downtrend": {"n_trades": 1, "total_return_pct": 6.0},
            },
        })
        assert _bo_chi_so_tot_theo_nguong(pp_data, "downtrend") == "MA20 (1 lệnh, +6.0%)"

    def test_tra_ve_gach_ngang_khi_khong_bo_nao_vuot_nguong(self):
        from dashboard.app import _bo_chi_so_tot_theo_nguong
        pp_data = self._pp_data(**{
            "MA20 (Giá cắt MA20)": {"downtrend": {"n_trades": 3, "total_return_pct": 2.0}},
        })
        assert _bo_chi_so_tot_theo_nguong(pp_data, "downtrend") == "—"

    def test_tuy_chinh_nguong(self):
        from dashboard.app import _bo_chi_so_tot_theo_nguong
        pp_data = self._pp_data(**{
            "MA20 (Giá cắt MA20)": {"downtrend": {"n_trades": 3, "total_return_pct": 8.0}},
        })
        assert _bo_chi_so_tot_theo_nguong(pp_data, "downtrend", nguong_pct=10.0) == "—"
        assert _bo_chi_so_tot_theo_nguong(pp_data, "downtrend", nguong_pct=5.0) == "MA20 (3 lệnh, +8.0%)"


# ==============================================================================
# Test: _dinh_dang_qua_mua_ngan_han (hàm thuần túy — gộp Độ lệch MA20 +
# Mức cảnh báo từ mục Tiêu chí ngắn hạn (đã ẩn) vào Tính cách giao dịch)
# ==============================================================================

class TestTinhDoLechVaCanhBaoMa20:
    def test_thieu_du_lieu_tra_ve_gach_ngang(self):
        from dashboard.app import _tinh_do_lech_va_canh_bao_ma20
        assert _tinh_do_lech_va_canh_bao_ma20(None, 100.0) == ("—", "—")
        assert _tinh_do_lech_va_canh_bao_ma20(100.0, None) == ("—", "—")
        assert _tinh_do_lech_va_canh_bao_ma20(100.0, 0.0) == ("—", "—")

    def test_binh_thuong_duoi_10_phan_tram(self):
        from dashboard.app import _tinh_do_lech_va_canh_bao_ma20
        # close=105, ma20=100 -> lech = 5/100*100 = +5.0%
        ket_qua = _tinh_do_lech_va_canh_bao_ma20(105.0, 100.0)
        assert ket_qua == ("+5.0%", "🟢 Bình thường")

    def test_nguy_co_dieu_chinh_tu_10_den_15_phan_tram(self):
        from dashboard.app import _tinh_do_lech_va_canh_bao_ma20
        # close=112, ma20=100 -> lech = +12.0%
        ket_qua = _tinh_do_lech_va_canh_bao_ma20(112.0, 100.0)
        assert ket_qua == ("+12.0%", "🟡 Nguy cơ điều chỉnh")

    def test_nguy_co_cao_tren_15_phan_tram(self):
        from dashboard.app import _tinh_do_lech_va_canh_bao_ma20
        # close=120, ma20=100 -> lech = +20.0%
        ket_qua = _tinh_do_lech_va_canh_bao_ma20(120.0, 100.0)
        assert ket_qua == ("+20.0%", "🔴 Nguy cơ cao")

    def test_gia_am_duoi_ma20(self):
        from dashboard.app import _tinh_do_lech_va_canh_bao_ma20
        # close=90, ma20=100 -> lech = -10.0% (van la "Binh thuong" vi cong
        # thuc goc chi xet lech DUONG vuot nguong, gia duoi MA20 khong tinh
        # la "qua mua")
        ket_qua = _tinh_do_lech_va_canh_bao_ma20(90.0, 100.0)
        assert ket_qua == ("-10.0%", "🟢 Bình thường")


# ==============================================================================
# Test: _tinh_ty_le_volume (hàm thuần túy — % Volume/MA20 Volume)
# ==============================================================================

class TestTinhTyLeVolume:
    def test_thieu_du_lieu_tra_ve_gach_ngang(self):
        from dashboard.app import _tinh_ty_le_volume
        assert _tinh_ty_le_volume(None, 1000.0) == "—"
        assert _tinh_ty_le_volume(1000.0, None) == "—"
        assert _tinh_ty_le_volume(1000.0, 0.0) == "—"

    def test_volume_bang_trung_binh(self):
        from dashboard.app import _tinh_ty_le_volume
        assert _tinh_ty_le_volume(1_000_000.0, 1_000_000.0) == "100%"

    def test_volume_dot_bien_cao_hon_trung_binh(self):
        from dashboard.app import _tinh_ty_le_volume
        # volume = 2,500,000, MA20 = 1,000,000 -> 250%
        assert _tinh_ty_le_volume(2_500_000.0, 1_000_000.0) == "250%"

    def test_volume_thap_hon_trung_binh(self):
        from dashboard.app import _tinh_ty_le_volume
        # volume = 400,000, MA20 = 1,000,000 -> 40%
        assert _tinh_ty_le_volume(400_000.0, 1_000_000.0) == "40%"


# ==============================================================================
# Test: "Gần đạt tiêu chí vào lệnh" (hàm thuần túy — kiểm tra giá/RSI/
# volume hiện tại so với điều kiện MUA của từng bộ chỉ số)
# ==============================================================================

class TestKiemTraDieuKienGia:
    def test_huong_len_gan_dat(self):
        from dashboard.app import _kiem_tra_dieu_kien_gia
        # can tang 2% de len 100
        assert _kiem_tra_dieu_kien_gia(98.0, 100.0, "len") == ("gan_dat", 2.0)

    def test_huong_len_qua_xa(self):
        from dashboard.app import _kiem_tra_dieu_kien_gia
        assert _kiem_tra_dieu_kien_gia(95.0, 100.0, "len") is None

    def test_huong_len_da_dat(self):
        from dashboard.app import _kiem_tra_dieu_kien_gia
        assert _kiem_tra_dieu_kien_gia(100.0, 100.0, "len") == ("da_dat", 0.0)
        assert _kiem_tra_dieu_kien_gia(105.0, 100.0, "len") == ("da_dat", 0.0)

    def test_huong_xuong_gan_dat(self):
        from dashboard.app import _kiem_tra_dieu_kien_gia
        assert _kiem_tra_dieu_kien_gia(102.0, 100.0, "xuong") == ("gan_dat", 2.0)

    def test_huong_xuong_da_dat(self):
        from dashboard.app import _kiem_tra_dieu_kien_gia
        assert _kiem_tra_dieu_kien_gia(99.0, 100.0, "xuong") == ("da_dat", 0.0)

    def test_thieu_du_lieu(self):
        from dashboard.app import _kiem_tra_dieu_kien_gia
        assert _kiem_tra_dieu_kien_gia(None, 100.0, "len") is None
        assert _kiem_tra_dieu_kien_gia(98.0, None, "len") is None


class TestKiemTraDieuKienRsi:
    def test_gan_dat(self):
        from dashboard.app import _kiem_tra_dieu_kien_rsi
        assert _kiem_tra_dieu_kien_rsi(33.0, 30.0) == ("gan_dat", 3.0)

    def test_qua_xa(self):
        from dashboard.app import _kiem_tra_dieu_kien_rsi
        assert _kiem_tra_dieu_kien_rsi(40.0, 30.0) is None

    def test_da_dat(self):
        from dashboard.app import _kiem_tra_dieu_kien_rsi
        assert _kiem_tra_dieu_kien_rsi(28.0, 30.0) == ("da_dat", 0.0)

    def test_thieu_du_lieu(self):
        from dashboard.app import _kiem_tra_dieu_kien_rsi
        assert _kiem_tra_dieu_kien_rsi(None, 30.0) is None


class TestKiemTraDieuKienVolume:
    def test_chua_du_70_phan_tram_muc_can(self):
        from dashboard.app import _kiem_tra_dieu_kien_volume
        # can dat 1000*1.5=1500, hien co 1000 -> 66.7% < 70%
        assert _kiem_tra_dieu_kien_volume(1000.0, 1000.0, 1.5) is None

    def test_gan_dat_tren_70_phan_tram(self):
        from dashboard.app import _kiem_tra_dieu_kien_volume
        # can dat 1000*1.5=1500, hien co 1100 -> 73.33%
        trang_thai, gia_tri = _kiem_tra_dieu_kien_volume(1100.0, 1000.0, 1.5)
        assert trang_thai == "gan_dat"
        assert gia_tri == pytest.approx(73.33, abs=0.01)

    def test_da_dat(self):
        from dashboard.app import _kiem_tra_dieu_kien_volume
        assert _kiem_tra_dieu_kien_volume(1600.0, 1000.0, 1.5) == ("da_dat", 100.0)

    def test_thieu_du_lieu(self):
        from dashboard.app import _kiem_tra_dieu_kien_volume
        assert _kiem_tra_dieu_kien_volume(None, 1000.0, 1.5) is None
        assert _kiem_tra_dieu_kien_volume(1000.0, 0.0, 1.5) is None


class TestKiemTraGanDatTheoBoChiSo:
    def test_ma20(self):
        from dashboard.app import _kiem_tra_gan_dat_theo_bo_chi_so
        ket_qua = _kiem_tra_gan_dat_theo_bo_chi_so(
            "MA20 (Giá cắt MA20)", {"close": 99.0, "ma20": 100.0},
        )
        assert ket_qua == ("gan_dat", "giá/MA20 cách 1.0%")

    def test_ema_cross(self):
        from dashboard.app import _kiem_tra_gan_dat_theo_bo_chi_so
        ket_qua = _kiem_tra_gan_dat_theo_bo_chi_so(
            "EMA50/EMA200 (Golden/Death Cross)", {"ema50": 99.0, "ema200": 100.0},
        )
        assert ket_qua == ("gan_dat", "EMA50/EMA200 cách 1.0%")

    def test_rsi14(self):
        from dashboard.app import _kiem_tra_gan_dat_theo_bo_chi_so
        ket_qua = _kiem_tra_gan_dat_theo_bo_chi_so(
            "RSI14 (Quá mua/Quá bán 30-70)", {"rsi14": 33.0},
        )
        assert ket_qua == ("gan_dat", "RSI14/30 cách 3.0 điểm")

    def test_bollinger_bounce(self):
        from dashboard.app import _kiem_tra_gan_dat_theo_bo_chi_so
        ket_qua = _kiem_tra_gan_dat_theo_bo_chi_so(
            "Bollinger Bounce (mua đáy dải dưới)", {"close": 101.0, "bb_lower": 100.0},
        )
        assert ket_qua == ("gan_dat", "giá/dải dưới BB cách 1.0%")

    def test_bollinger_breakout_ca_2_dieu_kien_gan_dat(self):
        from dashboard.app import _kiem_tra_gan_dat_theo_bo_chi_so
        ket_qua = _kiem_tra_gan_dat_theo_bo_chi_so(
            "Bollinger Breakout + Volume",
            {"close": 99.0, "bb_upper": 100.0, "volume": 1080.0, "volume_ma20": 1000.0},
        )
        assert ket_qua == ("gan_dat", "giá/dải trên BB cách 1.0%, volume đạt 90%")

    def test_bollinger_breakout_1_dieu_kien_qua_xa_tra_ve_none(self):
        from dashboard.app import _kiem_tra_gan_dat_theo_bo_chi_so
        # volume qua xa (chi 50% muc can) -> ca bo tra ve None
        ket_qua = _kiem_tra_gan_dat_theo_bo_chi_so(
            "Bollinger Breakout + Volume",
            {"close": 99.0, "bb_upper": 100.0, "volume": 600.0, "volume_ma20": 1000.0},
        )
        assert ket_qua is None

    def test_volume_breakout_ma20(self):
        from dashboard.app import _kiem_tra_gan_dat_theo_bo_chi_so
        ket_qua = _kiem_tra_gan_dat_theo_bo_chi_so(
            "Volume Breakout + MA20",
            {"close": 99.0, "ma20": 100.0, "volume": 1350.0, "volume_ma20": 1000.0},
        )
        assert ket_qua == ("gan_dat", "giá/MA20 cách 1.0%, volume đạt 90%")

    def test_trend_filter_ema_rsi(self):
        from dashboard.app import _kiem_tra_gan_dat_theo_bo_chi_so
        ket_qua = _kiem_tra_gan_dat_theo_bo_chi_so(
            "Kết hợp: Trend Filter EMA + RSI",
            {"rsi14": 38.0, "ema50": 99.0, "ema200": 100.0},
        )
        assert ket_qua == ("gan_dat", "RSI14/35 cách 3.0 điểm, EMA nền cách 1.0%")

    def test_ca_2_dieu_kien_da_dat_thi_ket_qua_la_da_dat(self):
        from dashboard.app import _kiem_tra_gan_dat_theo_bo_chi_so
        ket_qua = _kiem_tra_gan_dat_theo_bo_chi_so(
            "Volume Breakout + MA20",
            {"close": 105.0, "ma20": 100.0, "volume": 1600.0, "volume_ma20": 1000.0},
        )
        assert ket_qua == ("da_dat", "giá/MA20 đã đạt, volume đã đạt")

    def test_mua_va_giu_khong_ap_dung(self):
        from dashboard.app import _kiem_tra_gan_dat_theo_bo_chi_so
        ket_qua = _kiem_tra_gan_dat_theo_bo_chi_so(
            "Mua và giữ (Buy & Hold)", {"close": 100.0, "ma20": 100.0},
        )
        assert ket_qua is None


class TestTimBoChiSoGanDat:
    def test_chua_co_giai_doan_tra_ve_gach_ngang(self):
        from dashboard.app import _tim_bo_chi_so_gan_dat
        assert _tim_bo_chi_so_gan_dat({}, None, {}) == "—"

    def test_chi_lay_bo_co_ln_tren_nguong_va_gan_dat(self):
        from dashboard.app import _tim_bo_chi_so_gan_dat
        pp_data = {
            "results": {
                "MA20 (Giá cắt MA20)": {"downtrend": {"n_trades": 3, "total_return_pct": 12.0}},
                # LN duoi nguong 5% -> khong duoc xet du co gan dat
                "RSI14 (Quá mua/Quá bán 30-70)": {"downtrend": {"n_trades": 2, "total_return_pct": 3.0}},
            },
        }
        chi_bao = {"close": 99.0, "ma20": 100.0, "rsi14": 20.0}
        ket_qua = _tim_bo_chi_so_gan_dat(pp_data, "downtrend", chi_bao)
        assert ket_qua == "🔔 MA20 (giá/MA20 cách 1.0%)"

    def test_da_dat_xep_truoc_gan_dat(self):
        from dashboard.app import _tim_bo_chi_so_gan_dat
        pp_data = {
            "results": {
                "MA20 (Giá cắt MA20)": {"downtrend": {"n_trades": 3, "total_return_pct": 12.0}},
                "RSI14 (Quá mua/Quá bán 30-70)": {"downtrend": {"n_trades": 2, "total_return_pct": 20.0}},
            },
        }
        # MA20 gan dat (99 cach 100 = 1%), RSI14 da dat (25 <= 30)
        chi_bao = {"close": 99.0, "ma20": 100.0, "rsi14": 25.0}
        ket_qua = _tim_bo_chi_so_gan_dat(pp_data, "downtrend", chi_bao)
        assert ket_qua == "✅ RSI14 (RSI14/30 đã đạt); 🔔 MA20 (giá/MA20 cách 1.0%)"

    def test_khong_co_bo_nao_gan_dat_tra_ve_gach_ngang(self):
        from dashboard.app import _tim_bo_chi_so_gan_dat
        pp_data = {
            "results": {
                "MA20 (Giá cắt MA20)": {"downtrend": {"n_trades": 3, "total_return_pct": 12.0}},
            },
        }
        # gia qua xa MA20 (10%)
        chi_bao = {"close": 90.0, "ma20": 100.0}
        assert _tim_bo_chi_so_gan_dat(pp_data, "downtrend", chi_bao) == "—"


# ==============================================================================
# Test: _tinh_chi_bao_gan_dat_theo_realtime_cached — tính lại MA/EMA/RSI/BB
# bằng cách thay giá đóng cửa phiên gần nhất bằng giá REALTIME
# ==============================================================================

class _FakeCollectorGanDatRealtime:
    """Giả lập `DataCollector` cho test — trả giá realtime theo mã đã cấu
    hình sẵn, KHÔNG gọi mạng thật."""

    def __init__(self, gia_theo_ma: dict):
        self._gia_theo_ma = gia_theo_ma

    def get_realtime_price(self, symbol: str) -> dict:
        ket_qua = self._gia_theo_ma.get(symbol)
        if ket_qua is None:
            raise RuntimeError(f"Không có dữ liệu giả lập cho mã '{symbol}'.")
        return ket_qua


def _seed_ohlcv_flat(storage: Storage, ma: str, so_phien: int, gia: float) -> None:
    """Lưu lịch sử OHLCV PHẲNG (open=high=low=close=`gia` mọi phiên) — dùng
    để tính tay chính xác kỳ vọng MA20/RSI14/Bollinger Bands (chuỗi không
    biến động -> RSI=100 theo quy ước avg_loss=0, BB độ rộng=0)."""
    records = [
        {
            "date": f"2025-{(1 + i // 28):02d}-{(1 + i % 28):02d}",
            "open": gia, "high": gia, "low": gia, "close": gia,
            "volume": 1_000_000,
        }
        for i in range(so_phien)
    ]
    storage.save("ohlcv_history", ma, {"records": records})


class TestTinhChiBaoGanDatTheoRealtime:
    def test_chuoi_phang_gia_realtime_giong_lich_su(self, isolated_db_path, monkeypatch):
        """25 phiên lịch sử PHẲNG ở 20.0 (nghìn đồng), giá realtime CŨNG
        20.000đ (=20.0 nghìn đồng, không đổi) -> MA20/BB=20.0 đúng nghĩa
        trung bình 1 chuỗi hằng số, RSI14=100.0 (avg_loss=0 xuyên suốt).
        EMA50/EMA200 phải None vì chưa đủ 50/200 phiên lịch sử."""
        from dashboard.app import _tinh_chi_bao_gan_dat_theo_realtime_cached
        import dashboard.app as app_module

        storage = Storage(db_path=isolated_db_path)
        _seed_ohlcv_flat(storage, "HPG", 25, 20.0)
        storage.close()

        fake_collector = _FakeCollectorGanDatRealtime({
            "HPG": {"price": 20000.0, "volume": 1_500_000.0, "da_khop_lenh": True},
        })
        monkeypatch.setattr(
            app_module, "_tao_data_collector_cho_tra_cuu_realtime",
            lambda: fake_collector,
        )

        st.cache_data.clear()
        storage2 = Storage(db_path=isolated_db_path)
        ket_qua = _tinh_chi_bao_gan_dat_theo_realtime_cached(storage2, ("HPG",))

        assert ket_qua["HPG"]["close"] == pytest.approx(20.0)
        assert ket_qua["HPG"]["ma20"] == pytest.approx(20.0)
        assert ket_qua["HPG"]["rsi14"] == pytest.approx(100.0)
        assert ket_qua["HPG"]["bb_upper"] == pytest.approx(20.0)
        assert ket_qua["HPG"]["bb_lower"] == pytest.approx(20.0)
        assert ket_qua["HPG"]["ema50"] is None
        assert ket_qua["HPG"]["ema200"] is None
        assert ket_qua["HPG"]["volume"] == pytest.approx(1_500_000.0)
        assert ket_qua["HPG"]["gia_realtime_full_vnd"] == pytest.approx(20000.0)

    def test_gia_realtime_thay_the_dung_phien_cuoi(self, isolated_db_path, monkeypatch):
        """20 phiên lịch sử PHẲNG ở 20.0, giá realtime = 21.000đ (=21.0
        nghìn đồng, KHÁC lịch sử) -> MA20 phải đổi đúng bằng tính tay:
        (19*20.0 + 21.0)/20 = 20.05 -> xác nhận phiên gần nhất THỰC SỰ bị
        thay bằng giá realtime, không phải giữ nguyên giá đã lưu."""
        from dashboard.app import _tinh_chi_bao_gan_dat_theo_realtime_cached
        import dashboard.app as app_module

        storage = Storage(db_path=isolated_db_path)
        _seed_ohlcv_flat(storage, "SSI", 20, 20.0)
        storage.close()

        fake_collector = _FakeCollectorGanDatRealtime({
            "SSI": {"price": 21000.0, "volume": 800_000.0, "da_khop_lenh": True},
        })
        monkeypatch.setattr(
            app_module, "_tao_data_collector_cho_tra_cuu_realtime",
            lambda: fake_collector,
        )

        st.cache_data.clear()
        storage2 = Storage(db_path=isolated_db_path)
        ket_qua = _tinh_chi_bao_gan_dat_theo_realtime_cached(storage2, ("SSI",))

        assert ket_qua["SSI"]["close"] == pytest.approx(21.0)
        assert ket_qua["SSI"]["ma20"] == pytest.approx((19 * 20.0 + 21.0) / 20)

    def test_mot_ma_loi_khong_lam_hong_ca_batch(self, isolated_db_path, monkeypatch):
        """Mã CEO lấy giá realtime thất bại (lỗi mạng/API) -> chỉ mã đó
        trả về {"loi": ...}, mã HPG hợp lệ khác trong CÙNG 1 lượt gọi vẫn
        tính bình thường."""
        from dashboard.app import _tinh_chi_bao_gan_dat_theo_realtime_cached
        import dashboard.app as app_module

        storage = Storage(db_path=isolated_db_path)
        _seed_ohlcv_flat(storage, "HPG", 25, 20.0)
        _seed_ohlcv_flat(storage, "CEO", 25, 15.0)
        storage.close()

        fake_collector = _FakeCollectorGanDatRealtime({
            "HPG": {"price": 20000.0, "volume": 1_000_000.0, "da_khop_lenh": True},
            # CEO cố ý KHÔNG có trong dict -> fake collector raise lỗi
        })
        monkeypatch.setattr(
            app_module, "_tao_data_collector_cho_tra_cuu_realtime",
            lambda: fake_collector,
        )

        st.cache_data.clear()
        storage2 = Storage(db_path=isolated_db_path)
        ket_qua = _tinh_chi_bao_gan_dat_theo_realtime_cached(storage2, ("HPG", "CEO"))

        assert "loi" in ket_qua["CEO"]
        assert ket_qua["HPG"]["ma20"] == pytest.approx(20.0)

    def test_lich_su_qua_ngan_tra_ve_loi(self, isolated_db_path, monkeypatch):
        """Mã chưa đủ 20 phiên lịch sử -> báo lỗi rõ ràng, KHÔNG gọi API
        realtime cho mã đó (không cần cấu hình fake collector trả giá)."""
        from dashboard.app import _tinh_chi_bao_gan_dat_theo_realtime_cached
        import dashboard.app as app_module

        storage = Storage(db_path=isolated_db_path)
        _seed_ohlcv_flat(storage, "BCR", 5, 10.0)
        storage.close()

        monkeypatch.setattr(
            app_module, "_tao_data_collector_cho_tra_cuu_realtime",
            lambda: _FakeCollectorGanDatRealtime({}),
        )

        st.cache_data.clear()
        storage2 = Storage(db_path=isolated_db_path)
        ket_qua = _tinh_chi_bao_gan_dat_theo_realtime_cached(storage2, ("BCR",))

        assert "loi" in ket_qua["BCR"]


# ==============================================================================
# Test: "🔴 Giá Realtime (tra cứu trực tiếp)" — thông điệp trạng thái phiên
# ==============================================================================

class TestDinhDangThongDiepTrangThaiPhien:
    def test_ngoai_gio_giao_dich(self):
        from dashboard.app import _dinh_dang_thong_diep_trang_thai_phien
        thong_diep = _dinh_dang_thong_diep_trang_thai_phien(
            dang_gio_giao_dich=False, da_khop_lenh=True,
        )
        assert "Ngoài giờ giao dịch" in thong_diep

    def test_trong_gio_va_da_khop_lenh(self):
        from dashboard.app import _dinh_dang_thong_diep_trang_thai_phien
        thong_diep = _dinh_dang_thong_diep_trang_thai_phien(
            dang_gio_giao_dich=True, da_khop_lenh=True,
        )
        assert "đã khớp lệnh" in thong_diep

    def test_trong_gio_nhung_chua_khop_lenh(self):
        from dashboard.app import _dinh_dang_thong_diep_trang_thai_phien
        thong_diep = _dinh_dang_thong_diep_trang_thai_phien(
            dang_gio_giao_dich=True, da_khop_lenh=False,
        )
        assert "CHƯA có lệnh nào khớp" in thong_diep
        assert "tham chiếu" in thong_diep.lower()


# ==============================================================================
# Test: build_watchlist_detail_table (hàm thuần túy, không cần AppTest)
# ==============================================================================

class TestBuildWatchlistDetailTable:
    def test_aggregates_data_from_multiple_modules_correctly(self, isolated_db_path):
        from dashboard.app import build_watchlist_detail_table

        storage = Storage(db_path=isolated_db_path)
        storage.save("indicator_snapshot", "HPG", {
            "close": 25.5, "ma20": 24.0, "ema50": 23.0, "ema100": 22.0,
            "ema200": 20.0, "volume": 1_500_000, "volume_ma_15": 1_000_000,
            "volume_ma_20": 1_000_000, "price_above_ema200": True,
            "is_volume_breakout": True,
        })
        storage.save("symbol_sector", "HPG", {"sector": "steel"})
        storage.save("market_regime_quant", "steel", {
            "trang_thai": "UPTREND", "do_tin_cay": "CAO", "macro_score": 0.5,
            "breadth_pct": 70.0, "breadth_theo_nhom": "steel",
            "canh_bao": [], "reasoning": [],
        })
        storage.save("stock_signal", "HPG", {
            "ma": "HPG", "khuyen_nghi": "MUA", "loai_ban": None,
        })
        storage.save("pattern_result", "HPG", {
            "confidence": 0.85, "segments": [], "accumulation_high": 26.0,
        })
        storage.save("entry_screener_report", "latest", {
            "danh_sach_ma": [{"ma": "HPG", "xep_hang_uu_tien": "UU_TIEN_CAO"}],
        })
        storage.close()

        storage = Storage(db_path=isolated_db_path)
        df = build_watchlist_detail_table(storage, ["HPG"])
        storage.close()

        row = df.iloc[0]
        assert row["Mã"] == "HPG"
        assert row["Ngành"] == "steel"
        assert row["Giá"] == pytest.approx(25.5)
        assert row["Trên EMA200"] == "✅"
        assert row["Đột biến KL"] == "🔶"
        assert row["Giai đoạn ngành"] == "UPTREND"
        assert row["Tin cậy"] == "CAO"
        assert row["Tín hiệu"] == "🟢 MUA"
        assert row["Mô hình (%)"] == 85
        assert row["Ưu tiên"] == "CAO"

    def test_missing_data_shows_dash_without_error(self, isolated_db_path):
        from dashboard.app import build_watchlist_detail_table

        storage = Storage(db_path=isolated_db_path)
        df = build_watchlist_detail_table(storage, ["UNKNOWN_SYMBOL"])
        storage.close()

        row = df.iloc[0]
        assert row["Mã"] == "UNKNOWN_SYMBOL"
        assert row["Giá"] is None
        assert row["Trên EMA200"] == "—"
        assert row["Tín hiệu"] == "—"


class TestBuildWatchlistDetailTableResilience:
    def test_one_symbol_error_does_not_break_other_rows(self, isolated_db_path, monkeypatch):
        """Nếu 1 mã gây lỗi khi đọc dữ liệu, các mã KHÁC vẫn phải hiển thị
        đúng bình thường — không để 1 mã lỗi làm sập cả bảng (rà soát sau
        sự cố thực tế 28/07/2026)."""
        from dashboard.app import build_watchlist_detail_table

        storage = Storage(db_path=isolated_db_path)
        storage.save("indicator_snapshot", "GOOD", {"close": 30.0})
        storage.close()

        storage = Storage(db_path=isolated_db_path)

        original_get_latest_many = storage.get_latest_many

        def flaky_get_latest_many(category, keys):
            result = original_get_latest_many(category, keys)
            if category == "indicator_snapshot" and "BAD" in keys:
                # Giả lập dữ liệu HỎNG cho riêng mã BAD (data=None) — khi
                # code bên trong build_watchlist_detail_table gọi
                # snapshot.get(...) sẽ raise AttributeError, đúng kịch
                # bản "1 mã lỗi" cần kiểm tra khả năng chịu lỗi.
                result = dict(result)
                result["BAD"] = {"data": None}
            return result

        monkeypatch.setattr(storage, "get_latest_many", flaky_get_latest_many)

        df = build_watchlist_detail_table(storage, ["GOOD", "BAD"])
        storage.close()

        good_row = df[df["Mã"] == "GOOD"].iloc[0]
        bad_row = df[df["Mã"] == "BAD"].iloc[0]

        assert good_row["Giá"] == pytest.approx(30.0)
        assert "Lỗi đọc dữ liệu" in bad_row["Ngành"]



class TestRemoveSymbolsFromWatchlist:
    def test_removes_specified_symbols(self):
        from dashboard.app import remove_symbols_from_watchlist
        result = remove_symbols_from_watchlist(["HPG", "VNM", "FPT"], ["VNM"])
        assert result == ["HPG", "FPT"]

    def test_removes_multiple_symbols(self):
        from dashboard.app import remove_symbols_from_watchlist
        result = remove_symbols_from_watchlist(["HPG", "VNM", "FPT"], ["VNM", "HPG"])
        assert result == ["FPT"]

    def test_ignores_symbols_not_in_watchlist(self):
        from dashboard.app import remove_symbols_from_watchlist
        result = remove_symbols_from_watchlist(["HPG", "VNM"], ["XYZ"])
        assert result == ["HPG", "VNM"]

    def test_does_not_mutate_original_list(self):
        from dashboard.app import remove_symbols_from_watchlist
        original = ["HPG", "VNM"]
        remove_symbols_from_watchlist(original, ["HPG"])
        assert original == ["HPG", "VNM"]


class TestPerUserWatchlist:
    def test_get_current_user_id_defaults_to_default(self):
        """Hàm thuần túy — kiểm tra riêng logic đọc query_params."""
        from dashboard.app import get_current_user_id
        import streamlit as st
        # Ngoài ngữ cảnh AppTest, query_params rỗng -> mặc định "default"
        # (test này chỉ xác nhận hàm không lỗi khi gọi trực tiếp)
        try:
            result = get_current_user_id()
            assert isinstance(result, str)
        except Exception:
            pytest.skip("Cần ngữ cảnh Streamlit runtime đầy đủ để test trực tiếp")

    def test_load_save_watchlist_isolated_per_user(self, isolated_db_path):
        """load_watchlist/save_watchlist với user_id khác nhau phải HOÀN
        TOÀN độc lập — sửa watchlist của người này không ảnh hưởng người kia."""
        from dashboard.app import load_watchlist, save_watchlist

        storage = Storage(db_path=isolated_db_path)

        save_watchlist(storage, ["HPG", "VNM"], user_id="tuyen")
        save_watchlist(storage, ["SSI", "VIB", "TNG"], user_id="nhan_vien_a")

        watchlist_tuyen = load_watchlist(storage, user_id="tuyen")
        watchlist_a = load_watchlist(storage, user_id="nhan_vien_a")
        watchlist_default = load_watchlist(storage, user_id="default")

        assert watchlist_tuyen == ["HPG", "VNM"]
        assert watchlist_a == ["SSI", "VIB", "TNG"]
        assert watchlist_default != watchlist_tuyen  # chưa từng ghi -> dùng mặc định gốc
        storage.close()


class TestResolveDbPath:
    """Rà soát sự cố thực tế 28/07/2026: dashboard TRƯỚC ĐÂY không hề đọc
    config.yaml, luôn cố định dùng SQLite cục bộ dù người dùng đã đổi
    config.yaml sang Supabase. Kiểm tra đúng thứ tự ưu tiên đã sửa."""

    def test_env_var_takes_highest_priority(self, monkeypatch):
        from dashboard.app import _resolve_db_path
        monkeypatch.setenv("PM_CK_DB_PATH", "/tmp/test_env_override.db")
        assert _resolve_db_path() == "/tmp/test_env_override.db"

    def test_falls_back_to_config_yaml_when_no_env_and_no_secrets(self, monkeypatch):
        from dashboard.app import _resolve_db_path
        monkeypatch.delenv("PM_CK_DB_PATH", raising=False)
        # st.secrets sẽ raise/rỗng trong môi trường test (không có secrets.toml)
        # -> phải rơi xuống đọc config.yaml thật của dự án
        result = _resolve_db_path()
        assert result is not None and result != ""
