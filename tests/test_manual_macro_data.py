"""
Unit test cho core/manual_macro_data.py
"""

from __future__ import annotations

from datetime import date

import pytest

from core.manual_macro_data import (
    InvalidMacroSeriesError,
    add_cpi_us_entry,
    add_entry,
    build_full_macro_score_engine_input,
    build_macro_score_engine_input,
    compute_consecutive_increases,
    compute_delta_last,
    compute_distance_from_peak_pct,
    compute_ytd_change_pct,
    get_latest_cpi_us_yoy,
    get_recent_cpi_us_mom,
    remove_entry,
)


# ==============================================================================
# Test: add_entry
# ==============================================================================

class TestAddEntry:
    def test_adds_new_entry_sorted(self):
        series = []
        series = add_entry(series, date(2026, 1, 15), 5.25)
        series = add_entry(series, date(2026, 1, 1), 5.50)

        assert series[0]["date"] == "2026-01-01"
        assert series[1]["date"] == "2026-01-15"

    def test_overwrites_entry_on_same_date(self):
        series = add_entry([], date(2026, 1, 1), 5.50)
        series = add_entry(series, date(2026, 1, 1), 5.75)  # sửa lại giá trị ngày đó

        assert len(series) == 1
        assert series[0]["value"] == 5.75

    def test_does_not_mutate_original_list(self):
        original = [{"date": "2026-01-01", "value": 5.5}]
        add_entry(original, date(2026, 1, 15), 5.25)
        assert len(original) == 1  # không bị thêm vào danh sách gốc


# ==============================================================================
# Test: compute_delta_last
# ==============================================================================

class TestComputeDeltaLast:
    def test_known_delta(self):
        series = [
            {"date": "2026-01-01", "value": 5.50},
            {"date": "2026-03-15", "value": 5.25},
        ]
        assert compute_delta_last(series) == pytest.approx(-0.25)

    def test_none_for_insufficient_data(self):
        series = [{"date": "2026-01-01", "value": 5.50}]
        assert compute_delta_last(series) is None

    def test_none_for_empty_series(self):
        assert compute_delta_last([]) is None


# ==============================================================================
# Test: compute_ytd_change_pct
# ==============================================================================

class TestComputeYtdChangePct:
    def test_known_ytd_change(self):
        series = [
            {"date": "2026-01-05", "value": 24000},
            {"date": "2026-06-01", "value": 25200},
        ]
        result = compute_ytd_change_pct(series)
        assert result == pytest.approx((25200 - 24000) / 24000 * 100)

    def test_only_considers_entries_within_the_year(self):
        series = [
            {"date": "2025-12-01", "value": 30000},  # năm trước -> không tính
            {"date": "2026-01-10", "value": 24000},  # đầu năm nay
            {"date": "2026-06-01", "value": 25200},
        ]
        result = compute_ytd_change_pct(series)
        assert result == pytest.approx((25200 - 24000) / 24000 * 100)

    def test_none_for_empty_series(self):
        assert compute_ytd_change_pct([]) is None

    def test_raises_for_zero_start_value(self):
        series = [
            {"date": "2026-01-01", "value": 0},
            {"date": "2026-06-01", "value": 100},
        ]
        with pytest.raises(InvalidMacroSeriesError):
            compute_ytd_change_pct(series)


# ==============================================================================
# Test: compute_consecutive_increases
# ==============================================================================

class TestComputeConsecutiveIncreases:
    def test_known_streak(self):
        series = [
            {"date": "2026-01-01", "value": 100},
            {"date": "2026-01-08", "value": 98},   # giảm -> ngắt chuỗi tại đây
            {"date": "2026-01-15", "value": 99},   # tăng (1)
            {"date": "2026-01-22", "value": 100},  # tăng (2)
            {"date": "2026-01-29", "value": 101},  # tăng (3)
        ]
        assert compute_consecutive_increases(series) == 3

    def test_zero_when_latest_is_decrease(self):
        series = [
            {"date": "2026-01-01", "value": 100},
            {"date": "2026-01-08", "value": 95},
        ]
        assert compute_consecutive_increases(series) == 0

    def test_zero_for_insufficient_data(self):
        assert compute_consecutive_increases([{"date": "2026-01-01", "value": 100}]) == 0


# ==============================================================================
# Test: compute_distance_from_peak_pct
# ==============================================================================

class TestComputeDistanceFromPeakPct:
    def test_known_distance(self):
        series = [
            {"date": "2026-01-01", "value": 26000},  # đỉnh lịch sử
            {"date": "2026-06-01", "value": 24700},  # hiện tại
        ]
        result = compute_distance_from_peak_pct(series)
        assert result == pytest.approx((26000 - 24700) / 26000 * 100)

    def test_zero_when_latest_is_the_peak(self):
        series = [
            {"date": "2026-01-01", "value": 24000},
            {"date": "2026-06-01", "value": 26000},  # điểm mới nhất = đỉnh
        ]
        assert compute_distance_from_peak_pct(series) == pytest.approx(0.0)

    def test_none_for_empty_series(self):
        assert compute_distance_from_peak_pct([]) is None


# ==============================================================================
# Test: build_macro_score_engine_input
# ==============================================================================

class TestBuildMacroScoreEngineInput:
    def test_uses_real_computed_values_when_available(self):
        fed_series = [
            {"date": "2026-01-01", "value": 5.50},
            {"date": "2026-03-15", "value": 5.25},
        ]
        fx_series = [
            {"date": "2026-01-05", "value": 24000},
            {"date": "2026-06-01", "value": 25200},
        ]
        result = build_macro_score_engine_input(fed_series, fx_series)

        assert result["fed_rate_delta_last_meeting"] == pytest.approx(-0.25)
        assert result["fx_ytd_change_pct"] == pytest.approx((25200 - 24000) / 24000 * 100)

    def test_neutral_defaults_when_series_empty(self):
        result = build_macro_score_engine_input([], [])

        assert result["fed_rate_delta_last_meeting"] == 0.0
        assert result["fx_ytd_change_pct"] == 0.0
        assert result["fx_so_tuan_tang_lien_tiep"] == 0
        assert result["cpi_us_yoy"] == 2.0  # trung tính, đúng mục tiêu Fed
        assert result["cpi_vn_yoy"] == 4.0  # trung tính, đúng mục tiêu NHNN
        assert result["su_kien_hien_tai"] == "none"

    def test_result_is_valid_input_for_calculate_macro_score(self):
        # Đảm bảo output có thể đưa thẳng vào calculate_macro_score() mà không lỗi
        from core.macro_score_engine import calculate_macro_score

        fed_series = [
            {"date": "2026-01-01", "value": 5.50},
            {"date": "2026-03-15", "value": 5.25},
        ]
        fx_series = [{"date": "2026-01-05", "value": 24000}]

        macro_input = build_macro_score_engine_input(fed_series, fx_series)
        result = calculate_macro_score(macro_input)
        assert "macro_score" in result
        assert "nhan" in result


# ==============================================================================
# Test: add_cpi_us_entry / get_latest_cpi_us_yoy / get_recent_cpi_us_mom
# ==============================================================================

class TestCpiUsSeries:
    def test_add_and_retrieve_latest_yoy(self):
        series = add_cpi_us_entry([], date(2026, 1, 1), cpi_yoy=3.0, cpi_mom=0.2)
        series = add_cpi_us_entry(series, date(2026, 2, 1), cpi_yoy=2.8, cpi_mom=0.1)
        assert get_latest_cpi_us_yoy(series) == pytest.approx(2.8)

    def test_get_recent_mom_returns_last_n(self):
        series = []
        for i, (yoy, mom) in enumerate([(3.0, 0.3), (2.9, 0.2), (2.8, 0.1), (2.7, 0.0)]):
            series = add_cpi_us_entry(series, date(2026, i + 1, 1), cpi_yoy=yoy, cpi_mom=mom)
        result = get_recent_cpi_us_mom(series, n=3)
        assert result == [0.2, 0.1, 0.0]

    def test_overwrites_same_date(self):
        series = add_cpi_us_entry([], date(2026, 1, 1), cpi_yoy=3.0, cpi_mom=0.2)
        series = add_cpi_us_entry(series, date(2026, 1, 1), cpi_yoy=3.5, cpi_mom=0.4)
        assert len(series) == 1
        assert get_latest_cpi_us_yoy(series) == pytest.approx(3.5)

    def test_empty_series_returns_none_and_empty_list(self):
        assert get_latest_cpi_us_yoy([]) is None
        assert get_recent_cpi_us_mom([]) == []


# ==============================================================================
# Test: build_full_macro_score_engine_input
# ==============================================================================

class TestBuildFullMacroScoreEngineInput:
    def test_falls_back_to_neutral_when_nothing_extra_provided(self):
        result = build_full_macro_score_engine_input([], [])
        assert result["cpi_us_yoy"] == 2.0
        assert result["cpi_vn_yoy"] == 4.0
        assert result["su_kien_hien_tai"] == "none"

    def test_uses_real_cpi_us_data_when_provided(self):
        cpi_us_series = add_cpi_us_entry([], date(2026, 1, 1), cpi_yoy=3.2, cpi_mom=0.3)
        result = build_full_macro_score_engine_input([], [], cpi_us_series=cpi_us_series)
        assert result["cpi_us_yoy"] == pytest.approx(3.2)
        assert result["cpi_us_mom_3thang"] == [0.3]

    def test_uses_real_cpi_vn_data_when_provided(self):
        cpi_vn_series = add_entry([], date(2026, 1, 1), 5.0)
        result = build_full_macro_score_engine_input([], [], cpi_vn_series=cpi_vn_series)
        assert result["cpi_vn_yoy"] == pytest.approx(5.0)

    def test_custom_muc_tieu_cpi_vn_applied(self):
        result = build_full_macro_score_engine_input([], [], muc_tieu_cpi_vn=4.5)
        assert result["muc_tieu_cpi_vn"] == pytest.approx(4.5)

    def test_computes_interbank_spread_and_weekly_change(self):
        overnight = add_entry([], date(2026, 1, 1), 3.0)
        rate_3m = [
            {"date": "2026-01-01", "value": 5.0},
            {"date": "2026-01-08", "value": 5.5},
        ]
        result = build_full_macro_score_engine_input(
            [], [], interbank_overnight_series=overnight, interbank_3m_series=rate_3m,
        )
        assert result["interbank_do_doc_duong_cong"] == pytest.approx(5.5 - 3.0)
        assert result["interbank_thay_doi_tuan_3m"] == pytest.approx(0.5)

    def test_custom_event_key_applied(self):
        result = build_full_macro_score_engine_input([], [], event_key="conflict_outbreak")
        assert result["su_kien_hien_tai"] == "conflict_outbreak"

    def test_result_is_valid_input_for_calculate_macro_score(self):
        from core.macro_score_engine import calculate_macro_score

        cpi_us_series = add_cpi_us_entry([], date(2026, 1, 1), cpi_yoy=3.0, cpi_mom=0.2)
        overnight = add_entry([], date(2026, 1, 1), 3.0)
        rate_3m = add_entry([], date(2026, 1, 1), 5.0)

        macro_input = build_full_macro_score_engine_input(
            [], [], cpi_us_series=cpi_us_series,
            interbank_overnight_series=overnight, interbank_3m_series=rate_3m,
            event_key="de_escalation_signal",
        )
        result = calculate_macro_score(macro_input)
        assert "macro_score" in result


# ==============================================================================
# Test: remove_entry
# ==============================================================================

class TestRemoveEntry:
    def test_removes_entry_by_date(self):
        series = add_entry([], date(2026, 1, 1), 5.0)
        series = add_entry(series, date(2026, 2, 1), 5.5)

        result = remove_entry(series, date(2026, 1, 1))

        assert len(result) == 1
        assert result[0]["date"] == "2026-02-01"

    def test_does_not_error_when_date_not_found(self):
        series = add_entry([], date(2026, 1, 1), 5.0)
        result = remove_entry(series, date(2026, 12, 31))
        assert len(result) == 1  # không đổi gì

    def test_does_not_mutate_original_list(self):
        series = add_entry([], date(2026, 1, 1), 5.0)
        remove_entry(series, date(2026, 1, 1))
        assert len(series) == 1  # gốc không bị ảnh hưởng

    def test_works_on_cpi_us_style_entries_too(self):
        # remove_entry chỉ lọc theo "date", không quan tâm cấu trúc còn lại
        series = add_cpi_us_entry([], date(2026, 1, 1), cpi_yoy=3.0, cpi_mom=0.2)
        series = add_cpi_us_entry(series, date(2026, 2, 1), cpi_yoy=2.8, cpi_mom=0.1)
        result = remove_entry(series, date(2026, 1, 1))
        assert len(result) == 1
        assert result[0]["date"] == "2026-02-01"
