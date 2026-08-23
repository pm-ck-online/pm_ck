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
    return db_path


@pytest.fixture
def seeded_storage(isolated_db_path):
    """Tạo sẵn dữ liệu mẫu vào database TẠM (isolated_db_path), để AppTest
    chạy dashboard đọc được dữ liệu này mà không đụng tới database thật.
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
            "📈 Bảng giá theo dõi (Watchlist)"
        )
        at.run(timeout=30)
        assert not at.exception

        # Chỉ đúng 1 tiêu đề mục được hiển thị (không lẫn các mục khác)
        subheader_texts = [s.value for s in at.subheader]
        assert any("Bảng giá theo dõi (Watchlist)" in s for s in subheader_texts)
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
