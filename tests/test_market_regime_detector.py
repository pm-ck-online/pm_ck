"""
Unit test cho core/market_regime_detector.py

Theo đúng yêu cầu dự án: viết test cho TỪNG NGUYÊN TẮC RIÊNG LẺ (test
riêng bước vĩ mô, test riêng bước EMA200, test riêng độ trễ xác nhận)
TRƯỚC KHI test tổ hợp cả hệ thống.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from core.data_collector import MacroDataPoint
from core.market_regime_detector import (
    apply_confirmation_lag,
    calculate_macro_score,
    classify_sector_trend,
    detect_market_regime,
    detect_market_regime_quant,
    evaluate_macro_filter,
)


def _macro(category, direction, affected_sectors=None, description="mô tả giả lập"):
    return MacroDataPoint(
        category=category,
        description=description,
        direction=direction,
        affected_sectors=affected_sectors or [],
        as_of=datetime.now(),
    )


def _snapshot(close, ema200, above=None):
    """Tạo 1 snapshot chỉ báo giả lập cho 1 mã cổ phiếu."""
    if above is None:
        above = close > ema200 if ema200 is not None else None
    return {
        "close": close,
        "ema200": ema200,
        "price_above_ema200": above,
    }


# ==============================================================================
# TEST RIÊNG BƯỚC 1 — VĨ MÔ (evaluate_macro_filter)
# ==============================================================================

class TestEvaluateMacroFilter:
    def test_no_macro_data_returns_no_caution(self):
        result = evaluate_macro_filter([])
        assert result["caution_sectors"] == set()
        assert result["overall_bias"] == "neutral"

    def test_tightening_sector_policy_flags_correct_sector(self):
        macro = [
            _macro("sector_policy", "tightening", affected_sectors=["real_estate"]),
        ]
        result = evaluate_macro_filter(macro)
        assert "real_estate" in result["caution_sectors"]

    def test_easing_policy_does_not_flag_caution(self):
        macro = [
            _macro("interest_rate", "easing", affected_sectors=["banking"]),
        ]
        result = evaluate_macro_filter(macro)
        assert "banking" not in result["caution_sectors"]

    def test_multiple_sectors_flagged_from_multiple_points(self):
        macro = [
            _macro("sector_policy", "tightening", affected_sectors=["real_estate"]),
            _macro("interest_rate", "tightening", affected_sectors=["banking"]),
        ]
        result = evaluate_macro_filter(macro)
        assert result["caution_sectors"] == {"real_estate", "banking"}

    def test_overall_bias_tightening_when_majority_tightening(self):
        macro = [
            _macro("omo", "tightening"),
            _macro("interest_rate", "tightening"),
            _macro("fx_intervention", "easing"),
        ]
        result = evaluate_macro_filter(macro)
        assert result["overall_bias"] == "tightening"

    def test_overall_bias_easing_when_majority_easing(self):
        macro = [
            _macro("omo", "easing"),
            _macro("interest_rate", "easing"),
            _macro("fx_intervention", "tightening"),
        ]
        result = evaluate_macro_filter(macro)
        assert result["overall_bias"] == "easing"

    def test_reasoning_is_not_empty(self):
        macro = [_macro("omo", "tightening", affected_sectors=["banking"])]
        result = evaluate_macro_filter(macro)
        assert len(result["reasoning"]) > 0


# ==============================================================================
# TEST RIÊNG BƯỚC 2 — EMA200 THEO NGÀNH (classify_sector_trend)
# ==============================================================================

class TestClassifySectorTrend:
    def test_uptrend_when_majority_above_ema200(self):
        snapshots = [
            _snapshot(close=120, ema200=100),  # +20% -> above
            _snapshot(close=115, ema200=100),  # +15% -> above
            _snapshot(close=90, ema200=100),   # below
        ]
        result = classify_sector_trend(snapshots)
        assert result["raw_regime"] == "uptrend"

    def test_downtrend_when_majority_below_ema200(self):
        snapshots = [
            _snapshot(close=80, ema200=100),   # -20% -> below
            _snapshot(close=85, ema200=100),   # -15% -> below
            _snapshot(close=110, ema200=100),  # above
        ]
        result = classify_sector_trend(snapshots)
        assert result["raw_regime"] == "downtrend"

    def test_sideway_when_split_evenly(self):
        snapshots = [
            _snapshot(close=110, ema200=100),  # above
            _snapshot(close=90, ema200=100),   # below
        ]
        result = classify_sector_trend(snapshots)
        assert result["raw_regime"] == "sideway"

    def test_sideway_when_price_hovers_close_to_ema200_despite_majority_above(self):
        # Đa số trên EMA200 nhưng khoảng cách RẤT NHỎ (giá xoay quanh trục)
        snapshots = [
            _snapshot(close=101, ema200=100),   # +1%
            _snapshot(close=100.5, ema200=100), # +0.5%
            _snapshot(close=99, ema200=100),    # -1%
        ]
        result = classify_sector_trend(
            snapshots, config={"sideway_distance_threshold_pct": 3.0}
        )
        assert result["raw_regime"] == "sideway"

    def test_excludes_symbols_with_none_ema200(self):
        snapshots = [
            _snapshot(close=120, ema200=100),
            {"close": 50, "ema200": None, "price_above_ema200": None},  # chưa đủ dữ liệu
        ]
        result = classify_sector_trend(snapshots)
        assert result["n_symbols_considered"] == 1

    def test_no_valid_symbols_returns_sideway_with_zero_confidence(self):
        snapshots = [{"close": 50, "ema200": None, "price_above_ema200": None}]
        result = classify_sector_trend(snapshots)
        assert result["raw_regime"] == "sideway"
        assert result["confidence"] == 0.0
        assert result["n_symbols_considered"] == 0

    def test_confidence_higher_for_stronger_uptrend(self):
        strong_uptrend = [_snapshot(close=150, ema200=100) for _ in range(10)]
        weak_uptrend = (
            [_snapshot(close=120, ema200=100) for _ in range(6)]
            + [_snapshot(close=80, ema200=100) for _ in range(4)]
        )
        strong_result = classify_sector_trend(strong_uptrend)
        weak_result = classify_sector_trend(weak_uptrend)
        assert strong_result["confidence"] >= weak_result["confidence"]


# ==============================================================================
# TEST RIÊNG ĐỘ TRỄ XÁC NHẬN (apply_confirmation_lag)
# ==============================================================================

class TestApplyConfirmationLag:
    def test_confirms_when_stable_for_required_sessions(self):
        history = ["uptrend", "uptrend", "uptrend"]
        result = apply_confirmation_lag(history, confirmation_lag_sessions=3)
        assert result == "uptrend"

    def test_does_not_confirm_when_not_enough_history(self):
        history = ["uptrend", "uptrend"]
        result = apply_confirmation_lag(history, confirmation_lag_sessions=3)
        assert result is None

    def test_does_not_confirm_when_signal_still_fluctuating(self):
        history = ["downtrend", "uptrend", "uptrend"]
        result = apply_confirmation_lag(history, confirmation_lag_sessions=3)
        assert result is None

    def test_only_considers_most_recent_n_sessions(self):
        # 2 phiên đầu khác biệt không quan trọng, chỉ cần 3 phiên GẦN NHẤT giống nhau
        history = ["downtrend", "sideway", "uptrend", "uptrend", "uptrend"]
        result = apply_confirmation_lag(history, confirmation_lag_sessions=3)
        assert result == "uptrend"


# ==============================================================================
# TEST TỔ HỢP — detect_market_regime (kết hợp cả 3 bước)
# ==============================================================================

class TestDetectMarketRegimeCombined:
    def test_pure_technical_uptrend_without_macro_caution(self):
        macro = []  # không có tín hiệu vĩ mô tiêu cực nào
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(5)]

        result = detect_market_regime(macro, snapshots, sector_name="banking")

        assert result["regime"] == "uptrend"
        assert result["affected_sectors"] == []

    def test_macro_caution_downgrades_uptrend_to_sideway(self):
        # Kỹ thuật cho thấy Uptrend rõ rệt, NHƯNG vĩ mô đang siết ngành này
        # -> PHẢI bị hạ xuống Sideway (vĩ mô ưu tiên trước kỹ thuật).
        macro = [
            _macro("sector_policy", "tightening", affected_sectors=["real_estate"]),
        ]
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(5)]

        result = detect_market_regime(macro, snapshots, sector_name="real_estate")

        assert result["regime"] == "sideway"
        assert "real_estate" in result["affected_sectors"]
        assert any("THẬN TRỌNG" in r for r in result["reasoning"])

    def test_macro_caution_on_different_sector_does_not_affect_this_one(self):
        # Vĩ mô siết ngành bất động sản, nhưng đang xét ngành NGÂN HÀNG
        # -> không bị ảnh hưởng, vẫn giữ nguyên kết quả kỹ thuật.
        macro = [
            _macro("sector_policy", "tightening", affected_sectors=["real_estate"]),
        ]
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(5)]

        result = detect_market_regime(macro, snapshots, sector_name="banking")

        assert result["regime"] == "uptrend"
        # Nhưng vẫn thấy real_estate được liệt kê trong affected_sectors để
        # tham khảo tổng quan toàn thị trường.
        assert "real_estate" in result["affected_sectors"]

    def test_downtrend_not_affected_by_macro_caution_flag(self):
        # Macro caution chỉ chặn Uptrend, không ép Downtrend thành gì khác.
        macro = [
            _macro("sector_policy", "tightening", affected_sectors=["real_estate"]),
        ]
        snapshots = [_snapshot(close=70, ema200=100) for _ in range(5)]

        result = detect_market_regime(macro, snapshots, sector_name="real_estate")
        assert result["regime"] == "downtrend"

    def test_no_history_returns_immediate_result(self):
        macro = []
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(5)]
        result = detect_market_regime(macro, snapshots, raw_regime_history=None)
        assert result["regime"] in {"uptrend", "downtrend", "sideway"}

    def test_confirmation_lag_blocks_immediate_regime_change(self):
        macro = []
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(5)]  # -> uptrend hôm nay

        # Lịch sử 2 phiên gần nhất KHÔNG phải uptrend -> chưa đủ 3 phiên ổn định
        history = ["sideway", "sideway"]
        result = detect_market_regime(
            macro, snapshots, raw_regime_history=history,
            config={"confirmation_lag_sessions": 3},
        )
        assert result["regime"] is None  # chưa xác nhận được

    def test_confirmation_lag_confirms_after_stable_sessions(self):
        macro = []
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(5)]  # -> uptrend hôm nay

        # Lịch sử 2 phiên gần nhất ĐÃ là uptrend -> hôm nay là phiên thứ 3 liên tiếp
        history = ["uptrend", "uptrend"]
        result = detect_market_regime(
            macro, snapshots, raw_regime_history=history,
            config={"confirmation_lag_sessions": 3},
        )
        assert result["regime"] == "uptrend"

    def test_reasoning_contains_both_macro_and_technical_info(self):
        macro = [_macro("omo", "tightening", affected_sectors=["banking"])]
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(5)]
        result = detect_market_regime(macro, snapshots, sector_name="banking")

        combined_reasoning = " ".join(result["reasoning"])
        assert "Vĩ mô" in combined_reasoning
        assert "EMA200" in combined_reasoning


# ==============================================================================
# TEST BỔ SUNG — Mô hình 3 lớp định lượng (theo tài liệu kỹ thuật chi tiết)
# ==============================================================================

class TestCalculateMacroScore:
    def test_empty_returns_zero(self):
        assert calculate_macro_score([]) == 0.0

    def test_all_tightening_yields_negative_score(self):
        macro = [_macro("interest_rate", "tightening"), _macro("fx_intervention", "tightening")]
        score = calculate_macro_score(macro)
        assert score < 0

    def test_all_easing_yields_positive_score(self):
        macro = [_macro("interest_rate", "easing"), _macro("fx_intervention", "easing")]
        score = calculate_macro_score(macro)
        assert score > 0

    def test_mixed_neutral_yields_score_near_zero(self):
        macro = [_macro("omo", "neutral"), _macro("sector_policy", "neutral")]
        score = calculate_macro_score(macro)
        assert score == pytest.approx(0.0)

    def test_score_bounded_within_range(self):
        macro = [_macro("interest_rate", "tightening")] * 10
        score = calculate_macro_score(macro)
        assert -2.0 <= score <= 2.0


class TestDetectMarketRegimeQuant:
    def test_positive_macro_and_high_breadth_yields_uptrend(self):
        macro = []  # trung tính
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(10)]  # 100% trên EMA200
        result = detect_market_regime_quant(macro, snapshots, group_name="Ngân hàng")
        assert result["trang_thai"] == "UPTREND"
        assert result["breadth_pct"] == pytest.approx(100.0)

    def test_negative_macro_caps_uptrend_to_sideway(self):
        macro = [_macro("interest_rate", "tightening")]
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(10)]  # kỹ thuật rất tốt
        result = detect_market_regime_quant(macro, snapshots)
        assert result["trang_thai"] == "SIDEWAY"  # bị hạ trần dù kỹ thuật tốt
        assert result["macro_score"] < 0

    def test_low_breadth_yields_downtrend(self):
        macro = []
        snapshots = [_snapshot(close=80, ema200=100) for _ in range(10)]  # 0% trên EMA200
        result = detect_market_regime_quant(macro, snapshots)
        assert result["trang_thai"] == "DOWNTREND"

    def test_extreme_breadth_triggers_warning(self):
        macro = []
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(10)]  # 100% -> vùng cực đoan
        result = detect_market_regime_quant(macro, snapshots)
        assert any("CỰC ĐOAN" in w for w in result["canh_bao"])

    def test_layer3_full_agreement_yields_high_confidence(self):
        macro = []
        # 70% breadth (7/10 mã trên EMA200) -> uptrend THƯỜNG, không rơi vào vùng cực đoan
        snapshots = (
            [_snapshot(close=130, ema200=100) for _ in range(7)]
            + [_snapshot(close=80, ema200=100) for _ in range(3)]
        )
        layer3 = {"ma_cross": "golden_cross", "adx": 30.0}
        result = detect_market_regime_quant(macro, snapshots, layer3_indicators=layer3)
        assert result["do_tin_cay"] == "CAO"
        assert result["canh_bao"] == []

    def test_layer3_conflict_yields_low_confidence_with_warning(self):
        macro = []
        snapshots = (
            [_snapshot(close=130, ema200=100) for _ in range(7)]
            + [_snapshot(close=80, ema200=100) for _ in range(3)]
        )
        layer3 = {"ma_cross": "death_cross", "adx": 10.0}  # cả 2 đều MÂU THUẪN với uptrend
        result = detect_market_regime_quant(macro, snapshots, layer3_indicators=layer3)
        assert result["do_tin_cay"] == "THAP"
        assert len(result["canh_bao"]) == 2

    def test_no_layer3_data_yields_medium_confidence(self):
        macro = []
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(10)]
        result = detect_market_regime_quant(macro, snapshots)
        assert result["do_tin_cay"] == "TRUNG_BINH"

    def test_output_structure_matches_spec(self):
        macro = []
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(5)]
        result = detect_market_regime_quant(macro, snapshots, group_name="Chứng khoán")
        expected_keys = {
            "trang_thai", "do_tin_cay", "macro_score", "breadth_pct",
            "breadth_theo_nhom", "canh_bao", "reasoning",
        }
        assert expected_keys.issubset(result.keys())
        assert result["breadth_theo_nhom"] == "Chứng khoán"

    def test_precomputed_macro_score_overrides_internal_calculation(self):
        # macro_context rỗng (sẽ cho macro_score=0.0 nếu tự tính), nhưng
        # truyền precomputed_macro_score âm rõ rệt -> PHẢI dùng giá trị này
        macro = []
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(10)]  # kỹ thuật rất tốt

        result = detect_market_regime_quant(
            macro, snapshots, precomputed_macro_score=-1.5,
        )
        assert result["macro_score"] == pytest.approx(-1.5)
        # Vĩ mô âm -> phải hạ trần xuống sideway dù kỹ thuật 100% trên EMA200
        assert result["trang_thai"] == "SIDEWAY"
        assert any("macro_score_engine chi tiết" in r for r in result["reasoning"])

    def test_precomputed_macro_score_positive_allows_uptrend(self):
        macro = []
        snapshots = [_snapshot(close=130, ema200=100) for _ in range(7)] + [
            _snapshot(close=80, ema200=100) for _ in range(3)
        ]
        result = detect_market_regime_quant(
            macro, snapshots, precomputed_macro_score=1.2,
        )
        assert result["macro_score"] == pytest.approx(1.2)
        assert result["trang_thai"] == "UPTREND"
