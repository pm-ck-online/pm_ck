"""
Unit test cho core/macro_score_engine.py
"""

from __future__ import annotations

import pytest

from core.macro_score_engine import (
    DEFAULT_WEIGHTS,
    InvalidMacroScoreError,
    calculate_macro_score,
    calculate_score_cpi_us,
    calculate_score_cpi_vn,
    calculate_score_event,
    calculate_score_fed,
    calculate_score_fx,
    calculate_score_interbank,
    classify_macro_score,
    clip,
    f_direction,
    f_dotplot,
)


# ==============================================================================
# Test: clip
# ==============================================================================

class TestClip:
    def test_within_range_unchanged(self):
        assert clip(1.0) == 1.0

    def test_clips_above_max(self):
        assert clip(5.0) == 2.0

    def test_clips_below_min(self):
        assert clip(-5.0) == -2.0

    def test_custom_bounds(self):
        assert clip(3.0, lo=0.0, hi=2.0) == 2.0


# ==============================================================================
# Test: f_direction / f_dotplot
# ==============================================================================

class TestFDirectionAndDotplot:
    def test_direction_cut_yields_positive(self):
        assert f_direction(-0.25) == 2.0

    def test_direction_hike_yields_negative(self):
        assert f_direction(0.25) == -2.0

    def test_direction_hold_yields_zero(self):
        assert f_direction(0.0) == 0.0

    def test_dotplot_dovish_shift_yields_positive(self):
        assert f_dotplot(-0.1) == 2.0

    def test_dotplot_hawkish_shift_yields_negative(self):
        assert f_dotplot(0.1) == -2.0


# ==============================================================================
# Test: calculate_score_fed
# ==============================================================================

class TestCalculateScoreFed:
    def test_known_value_cut_but_hawkish_dotplot(self):
        # Ví dụ trong tài liệu: Fed cắt lãi suất nhưng dot-plot chuyển diều hâu
        score = calculate_score_fed(delta_rate_last=-0.25, delta_dot_plot=0.1)
        assert score == pytest.approx(0.4 * 2.0 + 0.6 * (-2.0))  # = -0.4

    def test_hold_and_neutral_dotplot_yields_zero(self):
        score = calculate_score_fed(delta_rate_last=0.0, delta_dot_plot=0.0)
        assert score == pytest.approx(0.0)

    def test_cut_and_dovish_dotplot_yields_max_positive(self):
        score = calculate_score_fed(delta_rate_last=-0.25, delta_dot_plot=-0.1)
        assert score == pytest.approx(2.0)  # a1*2 + a2*2 = 2.0


# ==============================================================================
# Test: calculate_score_cpi_us
# ==============================================================================

class TestCalculateScoreCpiUs:
    def test_known_value_above_target_with_momentum(self):
        score = calculate_score_cpi_us(cpi_yoy=3.0, cpi_mom_3thang=[0.3, 0.3, 0.3])
        # gap=(3.0-2.0)/2.0=0.5; momentum=0.3/0.3=1.0
        expected = -0.5 * 0.5 - 0.5 * 1.0
        assert score == pytest.approx(expected)

    def test_at_target_with_zero_momentum_yields_zero(self):
        score = calculate_score_cpi_us(cpi_yoy=2.0, cpi_mom_3thang=[0.0, 0.0, 0.0])
        assert score == pytest.approx(0.0)

    def test_raises_for_empty_momentum_list(self):
        with pytest.raises(InvalidMacroScoreError):
            calculate_score_cpi_us(cpi_yoy=3.0, cpi_mom_3thang=[])


# ==============================================================================
# Test: calculate_score_cpi_vn
# ==============================================================================

class TestCalculateScoreCpiVn:
    def test_known_value_above_target(self):
        score = calculate_score_cpi_vn(cpi_yoy_vn=5.0, muc_tieu_cpi_vn=4.0)
        assert score == pytest.approx(-0.5)  # -1.0 * clip(1.0/2.0) = -0.5

    def test_at_target_yields_zero(self):
        score = calculate_score_cpi_vn(cpi_yoy_vn=4.0, muc_tieu_cpi_vn=4.0)
        assert score == pytest.approx(0.0)

    def test_below_target_yields_positive(self):
        score = calculate_score_cpi_vn(cpi_yoy_vn=3.0, muc_tieu_cpi_vn=4.0)
        assert score > 0


# ==============================================================================
# Test: calculate_score_fx
# ==============================================================================

class TestCalculateScoreFx:
    def test_known_value(self):
        score = calculate_score_fx(
            fx_ytd_change_pct=6.0, fx_so_tuan_tang_lien_tiep=8, fx_khoang_cach_dinh_pct=0.5,
        )
        term1 = clip(6.0 / 3.0)          # clip(2.0) = 2.0
        term2 = clip(8 / 10, 0.0, 2.0)   # 0.8
        term3 = clip((2.0 - 0.5) / 2.0)  # 0.75
        expected = -0.4 * term1 - 0.3 * term2 - 0.3 * term3
        assert score == pytest.approx(expected)

    def test_far_from_peak_yields_favorable_contribution(self):
        # Cách xa đỉnh lịch sử -> an toàn hơn -> đóng góp DƯƠNG (thuận lợi)
        score = calculate_score_fx(
            fx_ytd_change_pct=0.0, fx_so_tuan_tang_lien_tiep=0, fx_khoang_cach_dinh_pct=10.0,
        )
        assert score > 0


# ==============================================================================
# Test: calculate_score_interbank
# ==============================================================================

class TestCalculateScoreInterbank:
    def test_known_value(self):
        score = calculate_score_interbank(
            interbank_do_doc_duong_cong=4.0, interbank_thay_doi_tuan_3m=0.6,
        )
        expected = -0.5 * clip(4.0 / 3.0) - 0.5 * clip(0.6 / 0.5)
        assert score == pytest.approx(expected)

    def test_flat_curve_no_change_yields_zero(self):
        score = calculate_score_interbank(
            interbank_do_doc_duong_cong=0.0, interbank_thay_doi_tuan_3m=0.0,
        )
        assert score == pytest.approx(0.0)


# ==============================================================================
# Test: calculate_score_event
# ==============================================================================

class TestCalculateScoreEvent:
    def test_no_event(self):
        assert calculate_score_event("none") == 0.0

    def test_conflict_outbreak(self):
        assert calculate_score_event("conflict_outbreak") == -2.0

    def test_positive_resolution(self):
        assert calculate_score_event("positive_resolution") == 2.0

    def test_raises_for_invalid_event_key(self):
        with pytest.raises(InvalidMacroScoreError):
            calculate_score_event("khong_ton_tai")


# ==============================================================================
# Test: classify_macro_score
# ==============================================================================

class TestClassifyMacroScore:
    def test_tich_cuc(self):
        assert classify_macro_score(0.6, score_event=0.0) == "TICH_CUC"

    def test_trung_tinh(self):
        assert classify_macro_score(0.0, score_event=0.0) == "TRUNG_TINH"

    def test_tieu_cuc(self):
        assert classify_macro_score(-0.7, score_event=0.0) == "TIEU_CUC"

    def test_tieu_cuc_manh_requires_both_conditions(self):
        assert classify_macro_score(-1.5, score_event=-2.0) == "TIEU_CUC_MANH"

    def test_exactly_minus_one_with_severe_event_is_not_manh(self):
        # Biên: macro_score = ĐÚNG -1.0 (không nhỏ hơn strict) -> KHÔNG phải MẠNH
        # dù score_event rất xấu -- đúng theo điều kiện "< -1.0" (nghiêm ngặt)
        assert classify_macro_score(-1.0, score_event=-2.0) == "TIEU_CUC"

    def test_very_negative_but_event_not_severe_is_tieu_cuc_not_manh(self):
        # macro_score < -1.0 nhưng event KHÔNG đủ nghiêm trọng (> -1.5) -> chỉ TIEU_CUC
        assert classify_macro_score(-1.5, score_event=-1.0) == "TIEU_CUC"


# ==============================================================================
# Test: calculate_macro_score — hàm tổng hợp chính
# ==============================================================================

class TestCalculateMacroScore:
    def _base_data(self, **overrides):
        data = {
            "fed_rate_delta_last_meeting": 0.0,
            "fed_dotplot_delta": 0.0,
            "cpi_us_yoy": 2.0,
            "cpi_us_mom_3thang": [0.0, 0.0, 0.0],
            "cpi_vn_yoy": 4.0,
            "fx_ytd_change_pct": 0.0,
            "fx_so_tuan_tang_lien_tiep": 0,
            "fx_khoang_cach_dinh_pct": 2.0,
            "interbank_do_doc_duong_cong": 0.0,
            "interbank_thay_doi_tuan_3m": 0.0,
            "su_kien_hien_tai": "none",
        }
        data.update(overrides)
        return data

    def test_all_neutral_yields_trung_tinh(self):
        result = calculate_macro_score(self._base_data())
        assert result["nhan"] == "TRUNG_TINH"
        assert result["macro_score"] == pytest.approx(0.0, abs=0.05)

    def test_all_favorable_yields_tich_cuc(self):
        data = self._base_data(
            fed_rate_delta_last_meeting=-0.25, fed_dotplot_delta=-0.1,
            cpi_us_yoy=1.5, cpi_us_mom_3thang=[-0.1, -0.1, -0.1],
            cpi_vn_yoy=3.0,
            fx_ytd_change_pct=-2.0, fx_so_tuan_tang_lien_tiep=0, fx_khoang_cach_dinh_pct=10.0,
            interbank_do_doc_duong_cong=-1.0, interbank_thay_doi_tuan_3m=-0.3,
            su_kien_hien_tai="positive_resolution",
        )
        result = calculate_macro_score(data)
        assert result["nhan"] == "TICH_CUC"
        assert result["macro_score"] >= 0.5

    def test_severe_event_overrides_score_to_at_most_minus_one(self):
        data = self._base_data(su_kien_hien_tai="conflict_outbreak")
        result = calculate_macro_score(data)
        assert result["macro_score"] <= -1.0

    def test_extremely_negative_inputs_yield_tieu_cuc_manh(self):
        data = self._base_data(
            fed_rate_delta_last_meeting=0.5, fed_dotplot_delta=0.5,
            cpi_us_yoy=6.0, cpi_us_mom_3thang=[1.0, 1.0, 1.0],
            cpi_vn_yoy=10.0,
            fx_ytd_change_pct=10.0, fx_so_tuan_tang_lien_tiep=20, fx_khoang_cach_dinh_pct=0.0,
            interbank_do_doc_duong_cong=10.0, interbank_thay_doi_tuan_3m=2.0,
            su_kien_hien_tai="conflict_outbreak",
        )
        result = calculate_macro_score(data)
        assert result["nhan"] == "TIEU_CUC_MANH"
        assert result["macro_score"] < -1.0

    def test_sub_scores_detail_present_in_output(self):
        result = calculate_macro_score(self._base_data())
        expected_keys = {"fed", "cpi_us", "cpi_vn", "fx", "interbank", "event"}
        assert expected_keys.issubset(result["chi_tiet_sub_scores"].keys())

    def test_raises_for_missing_required_field(self):
        data = self._base_data()
        del data["cpi_vn_yoy"]
        with pytest.raises(InvalidMacroScoreError):
            calculate_macro_score(data)

    def test_custom_weights_override_default(self):
        data = self._base_data(su_kien_hien_tai="positive_resolution")
        custom_weights = {**DEFAULT_WEIGHTS, "event": 1.0, "fed": 0.0, "cpi_us": 0.0,
                          "cpi_vn": 0.0, "fx": 0.0, "interbank": 0.0}
        result = calculate_macro_score(data, weights=custom_weights)
        # Với trọng số event=1.0 và các trọng số khác=0 -> macro_score = score_event = 2.0
        assert result["macro_score"] == pytest.approx(2.0)

    def test_optional_muc_tieu_cpi_vn_can_be_overridden(self):
        data = self._base_data(cpi_vn_yoy=5.0, muc_tieu_cpi_vn=5.0)
        result = calculate_macro_score(data)
        # CPI VN đúng bằng mục tiêu tùy chỉnh -> sub-score cpi_vn phải = 0
        assert result["chi_tiet_sub_scores"]["cpi_vn"] == pytest.approx(0.0)
