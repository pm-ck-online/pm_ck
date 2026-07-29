"""
Unit test cho core/capital_allocation_engine.py

Bao gồm test tái hiện lại VÍ DỤ CTG trong tài liệu kỹ thuật gốc (mục 5) để
xác nhận công thức cài đặt đúng tinh thần tài liệu.
"""

from __future__ import annotations

import pytest

from core.capital_allocation_engine import (
    ALLOCATION_TABLE,
    InvalidCapitalAllocationError,
    allocate_capital_by_breadth,
    calculate_capital_allocation,
    calculate_entry_price_range,
    calculate_position_size,
    calculate_stock_allocation_pct,
    calculate_stop_loss_range,
    calculate_take_profit_range,
    find_support_resistance,
    round_to_lot,
)


# ==============================================================================
# Test: find_support_resistance
# ==============================================================================

class TestFindSupportResistance:
    def test_known_min_max_over_lookback(self):
        import pandas as pd
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=10),
            "open": [100] * 10,
            "high": [105, 110, 108, 120, 112, 115, 118, 111, 116, 114],
            "low": [95, 90, 98, 100, 97, 102, 99, 103, 101, 104],
            "close": [100] * 10,
            "volume": [1000] * 10,
        })
        support, resistance = find_support_resistance(df, lookback=10)
        assert support == pytest.approx(90.0)
        assert resistance == pytest.approx(120.0)

    def test_respects_lookback_window(self):
        import pandas as pd
        # 5 phiên đầu có giá cực trị lớn, 5 phiên sau ổn định hơn
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=10),
            "open": [100] * 10,
            "high": [200, 200, 200, 200, 200, 110, 111, 112, 113, 114],
            "low": [10, 10, 10, 10, 10, 95, 96, 97, 98, 99],
            "close": [100] * 10,
            "volume": [1000] * 10,
        })
        support, resistance = find_support_resistance(df, lookback=5)
        # Chỉ xét 5 phiên GẦN NHẤT -> bỏ qua các cực trị 200/10 ở đầu
        assert resistance == pytest.approx(114.0)
        assert support == pytest.approx(95.0)

    def test_raises_for_empty_df(self):
        import pandas as pd
        df = pd.DataFrame({"date": [], "open": [], "high": [], "low": [], "close": [], "volume": []})
        with pytest.raises(InvalidCapitalAllocationError):
            find_support_resistance(df)


# ==============================================================================
# Test: calculate_stock_allocation_pct
# ==============================================================================

class TestCalculateStockAllocationPct:
    def test_uptrend_high_confidence_yields_upper_bound(self):
        pct = calculate_stock_allocation_pct("UPTREND", "CAO")
        assert pct == pytest.approx(1.00)

    def test_uptrend_low_confidence_yields_lower_bound(self):
        pct = calculate_stock_allocation_pct("UPTREND", "THAP")
        assert pct == pytest.approx(0.70)

    def test_uptrend_medium_confidence_yields_midpoint(self):
        pct = calculate_stock_allocation_pct("UPTREND", "TRUNG_BINH")
        assert pct == pytest.approx(0.85)

    def test_downtrend_range(self):
        pct = calculate_stock_allocation_pct("DOWNTREND", "CAO")
        assert pct == pytest.approx(0.30)

    def test_raises_for_invalid_trang_thai(self):
        with pytest.raises(InvalidCapitalAllocationError):
            calculate_stock_allocation_pct("KHONGHOP LE", "CAO")

    def test_raises_for_invalid_do_tin_cay(self):
        with pytest.raises(InvalidCapitalAllocationError):
            calculate_stock_allocation_pct("UPTREND", "SIEU_CAO")


# ==============================================================================
# Test: calculate_entry_price_range — tái hiện đúng ví dụ CTG
# ==============================================================================

class TestCalculateEntryPriceRange:
    def test_ctg_pullback_example_from_spec(self):
        # Ví dụ CTG trong tài liệu: giá tham chiếu 42.000, ATR14=900, pullback
        low, high = calculate_entry_price_range(42000, 900, strategy="pullback")
        assert low == pytest.approx(41460)
        assert high == pytest.approx(41910)

    def test_breakout_range_above_reference(self):
        low, high = calculate_entry_price_range(100.0, 10.0, strategy="breakout")
        assert low == pytest.approx(100.5)
        assert high == pytest.approx(104.0)

    def test_support_range_centered_on_support_level(self):
        low, high = calculate_entry_price_range(
            100.0, 10.0, strategy="support", support_level=95.0
        )
        assert low == pytest.approx(93.0)
        assert high == pytest.approx(97.0)

    def test_raises_for_invalid_strategy(self):
        with pytest.raises(InvalidCapitalAllocationError):
            calculate_entry_price_range(100.0, 10.0, strategy="random")

    def test_raises_for_non_positive_reference_price(self):
        with pytest.raises(InvalidCapitalAllocationError):
            calculate_entry_price_range(0, 10.0)


# ==============================================================================
# Test: calculate_stop_loss_range / calculate_take_profit_range
# ==============================================================================

class TestStopLossAndTakeProfitRange:
    def test_ctg_stop_loss_example(self):
        low, high = calculate_stop_loss_range(support_level=41000, atr14=900)
        assert high == pytest.approx(41000)
        assert low == pytest.approx(41000 - 0.2 * 900)

    def test_take_profit_range(self):
        low, high = calculate_take_profit_range(resistance_level=44500, atr14=900)
        assert low == pytest.approx(44500)
        assert high == pytest.approx(44500 + 0.3 * 900)


# ==============================================================================
# Test: round_to_lot
# ==============================================================================

class TestRoundToLot:
    def test_rounds_down_to_nearest_lot(self):
        assert round_to_lot(4527, lot_size=100) == 4500

    def test_below_one_lot_returns_zero(self):
        assert round_to_lot(50, lot_size=100) == 0

    def test_exact_lot_unchanged(self):
        assert round_to_lot(4500, lot_size=100) == 4500


# ==============================================================================
# Test: calculate_position_size — tái hiện đúng ví dụ CTG (KL cuối = 4.500)
# ==============================================================================

class TestCalculatePositionSize:
    def test_ctg_example_final_quantity_matches_spec(self):
        # Toàn bộ số liệu lấy từ ví dụ CTG trong tài liệu (mục 5, Bước 4)
        entry_range = (41460, 41910)
        stop_loss_range = (41000 - 0.2 * 900, 41000)
        von_ma = 188_700_000

        qty = calculate_position_size(
            nav=2_000_000_000, risk_per_trade_pct=0.02,
            entry_price_range=entry_range, stop_loss_range=stop_loss_range,
            capital_budget=von_ma, lot_size=100,
        )
        # Tài liệu tính ra 4.500 cổ phiếu sau khi làm tròn lô — công thức
        # của mình cho kết quả cùng khoảng (chênh lệch nhỏ do làm tròn
        # trong ví dụ minh họa của tài liệu), nhưng SAU KHI LÀM TRÒN LÔ
        # phải khớp chính xác 4.500 như tài liệu.
        assert qty == 4500

    def test_limited_by_risk_when_budget_very_large(self):
        qty = calculate_position_size(
            nav=1_000_000, risk_per_trade_pct=0.02,
            entry_price_range=(110, 110), stop_loss_range=(100, 100),
            capital_budget=100_000_000,  # ngân sách vô cùng lớn -> bị chặn bởi rủi ro
        )
        # risk_amount=20,000; risk_per_share=10 -> qty=2000 -> lô tròn 2000
        assert qty == 2000

    def test_limited_by_budget_when_small(self):
        qty = calculate_position_size(
            nav=1_000_000_000, risk_per_trade_pct=0.02,
            entry_price_range=(110, 110), stop_loss_range=(100, 100),
            capital_budget=1_100,  # chỉ đủ mua 10 cổ phiếu
        )
        assert qty == 0  # 10 cổ phiếu < 1 lô (100) -> làm tròn về 0

    def test_raises_when_entry_not_above_stop_loss(self):
        with pytest.raises(InvalidCapitalAllocationError):
            calculate_position_size(
                nav=1_000_000, risk_per_trade_pct=0.02,
                entry_price_range=(90, 90), stop_loss_range=(100, 100),
            )


# ==============================================================================
# Test: allocate_capital_by_breadth
# ==============================================================================

class TestAllocateCapitalByBreadth:
    def test_ctg_ssi_nlg_example_from_spec(self):
        # Ví dụ tài liệu: breadth 68/60/55 -> tỷ lệ ~37%/33%/30%
        result = allocate_capital_by_breadth(
            510_000_000, {"CTG": 68, "SSI": 60, "NLG": 55}
        )
        assert result["CTG"] == pytest.approx(510_000_000 * 68 / 183, rel=0.01)
        assert result["SSI"] == pytest.approx(510_000_000 * 60 / 183, rel=0.01)
        assert result["NLG"] == pytest.approx(510_000_000 * 55 / 183, rel=0.01)

    def test_zero_breadth_splits_evenly(self):
        result = allocate_capital_by_breadth(100_000, {"A": 0, "B": 0})
        assert result["A"] == pytest.approx(50_000)
        assert result["B"] == pytest.approx(50_000)

    def test_raises_for_empty_input(self):
        with pytest.raises(InvalidCapitalAllocationError):
            allocate_capital_by_breadth(100_000, {})


# ==============================================================================
# Test: calculate_capital_allocation — hàm tổng hợp chính
# ==============================================================================

class TestCalculateCapitalAllocation:
    def _sample_watchlist(self):
        return [
            {
                "ma": "CTG", "nganh": "Ngân hàng", "atr14": 900,
                "gia_tham_chieu": 42000, "chien_luoc": "pullback",
                "ho_tro": 41000, "khang_cu": 44500,
            },
            {
                "ma": "SSI", "nganh": "Chứng khoán", "atr14": 500,
                "gia_tham_chieu": 30000, "chien_luoc": "breakout",
                "ho_tro": 29000, "khang_cu": 32000,
            },
        ]

    def test_uptrend_produces_three_tranches(self):
        result = calculate_capital_allocation(
            trang_thai="UPTREND", do_tin_cay="CAO",
            breadth_theo_nganh={"Ngân hàng": 68, "Chứng khoán": 60},
            nav=2_000_000_000, watchlist=self._sample_watchlist(),
        )
        assert len(result["cac_dot_giai_ngan"]) == 3
        assert result["cac_dot_giai_ngan"][0]["ty_le_dot"] == pytest.approx(0.30)

    def test_sideway_produces_two_tranches(self):
        result = calculate_capital_allocation(
            trang_thai="SIDEWAY", do_tin_cay="TRUNG_BINH",
            breadth_theo_nganh={"Giá trị / Cổ tức cao": 50},
            nav=1_000_000_000,
            watchlist=[{
                "ma": "REE", "nganh": "Giá trị / Cổ tức cao", "atr14": 800,
                "gia_tham_chieu": 60000, "chien_luoc": "support",
                "ho_tro": 58000, "khang_cu": 63000,
            }],
        )
        assert len(result["cac_dot_giai_ngan"]) == 2

    def test_downtrend_without_divergence_blocks_all_allocation(self):
        watchlist = [{
            "ma": "PNJ", "nganh": "Tiêu dùng thiết yếu", "atr14": 700,
            "gia_tham_chieu": 90000, "chien_luoc": "support",
            "ho_tro": 88000, "khang_cu": 95000,
            "co_phan_ky_tang": False,
        }]
        result = calculate_capital_allocation(
            trang_thai="DOWNTREND", do_tin_cay="TRUNG_BINH",
            breadth_theo_nganh={"Tiêu dùng thiết yếu": 35},
            nav=1_000_000_000, watchlist=watchlist,
        )
        assert result["ty_trong_co_phieu_khuyen_nghi"] == 0.0
        assert result["cac_dot_giai_ngan"] == []
        assert "KHÔNG GIẢI NGÂN" in result["canh_bao"][0]

    def test_downtrend_with_divergence_allows_allocation(self):
        watchlist = [{
            "ma": "PNJ", "nganh": "Tiêu dùng thiết yếu", "atr14": 700,
            "gia_tham_chieu": 90000, "chien_luoc": "support",
            "ho_tro": 88000, "khang_cu": 95000,
            "co_phan_ky_tang": True,
        }]
        result = calculate_capital_allocation(
            trang_thai="DOWNTREND", do_tin_cay="TRUNG_BINH",
            breadth_theo_nganh={"Tiêu dùng thiết yếu": 35},
            nav=1_000_000_000, watchlist=watchlist,
        )
        assert result["ty_trong_co_phieu_khuyen_nghi"] > 0
        assert len(result["cac_dot_giai_ngan"]) == 1  # downtrend -> 1 đợt duy nhất

    def test_no_symbol_matches_priority_sector_still_allocates_with_warning(self):
        watchlist = [{
            "ma": "XYZ", "nganh": "Công nghệ", "atr14": 500,
            "gia_tham_chieu": 50000, "chien_luoc": "breakout",
            "ho_tro": 48000, "khang_cu": 55000,
        }]
        result = calculate_capital_allocation(
            trang_thai="UPTREND", do_tin_cay="CAO",
            breadth_theo_nganh={"Công nghệ": 50},
            nav=1_000_000_000, watchlist=watchlist,
        )
        assert any("ngành ưu tiên" in w for w in result["canh_bao"])
        assert len(result["cac_dot_giai_ngan"][0]["danh_sach_ma"]) == 1

    def test_portfolio_risk_warning_when_exceeding_threshold(self):
        # Kịch bản CỐ TÌNH có biên độ cắt lỗ RỘNG (tương đối so với giá) để
        # khối lượng bị giới hạn bởi RỦI RO (không phải ngân sách vốn) —
        # kết hợp risk_per_trade_pct cao -> tổng rủi ro vượt ngưỡng 20%.
        watchlist = [{
            "ma": "TEST", "nganh": "Ngân hàng", "atr14": 60,
            "gia_tham_chieu": 100, "chien_luoc": "breakout",
            "ho_tro": 50, "khang_cu": 130,
        }]
        result = calculate_capital_allocation(
            trang_thai="UPTREND", do_tin_cay="CAO",
            breadth_theo_nganh={"Ngân hàng": 100},
            nav=2_000_000_000, watchlist=watchlist,
            risk_per_trade_pct=0.15,  # 15%/lệnh, cố tình rất cao để kiểm tra cảnh báo
        )
        assert any("VƯỢT" in w for w in result["canh_bao"])

    def test_raises_for_non_positive_nav(self):
        with pytest.raises(InvalidCapitalAllocationError):
            calculate_capital_allocation(
                trang_thai="UPTREND", do_tin_cay="CAO",
                breadth_theo_nganh={}, nav=0, watchlist=self._sample_watchlist(),
            )

    def test_raises_for_empty_watchlist(self):
        with pytest.raises(InvalidCapitalAllocationError):
            calculate_capital_allocation(
                trang_thai="UPTREND", do_tin_cay="CAO",
                breadth_theo_nganh={}, nav=1_000_000_000, watchlist=[],
            )

    def test_output_structure_matches_spec(self):
        result = calculate_capital_allocation(
            trang_thai="UPTREND", do_tin_cay="CAO",
            breadth_theo_nganh={"Ngân hàng": 68, "Chứng khoán": 60},
            nav=2_000_000_000, watchlist=self._sample_watchlist(),
        )
        expected_keys = {
            "nav_mo_phong", "trang_thai_thi_truong", "ty_trong_co_phieu_khuyen_nghi",
            "von_tien_mat_du_phong", "cac_dot_giai_ngan",
            "tong_rui_ro_danh_muc_hien_tai_pct", "canh_bao", "ghi_chu",
        }
        assert expected_keys.issubset(result.keys())

        first_symbol = result["cac_dot_giai_ngan"][0]["danh_sach_ma"][0]
        expected_symbol_keys = {
            "ma", "nganh", "von_phan_bo", "khoang_gia_vao_lenh",
            "khoi_luong_du_kien", "khoang_cat_lo", "khoang_chot_loi_tham_khao",
            "ty_le_rui_ro_tren_nav",
        }
        assert expected_symbol_keys.issubset(first_symbol.keys())
