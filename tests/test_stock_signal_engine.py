"""
Unit test cho core/stock_signal_engine.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.stock_signal_engine import (
    FUNDAMENTAL_BUY_THRESHOLDS,
    InvalidStockSignalError,
    build_signal_summary_report,
    check_base_trend_condition,
    check_breakout_pattern,
    check_buy_veto,
    check_ema200_break_confirmed,
    check_pullback_pattern,
    check_resistance_overbought,
    check_stop_loss_hit,
    check_support_bounce_pattern,
    check_volume_depletion,
    evaluate_fundamental_buy_screen,
    evaluate_fundamental_sell_screen,
    evaluate_stock_signal,
    evaluate_technical_buy_trigger,
    evaluate_technical_sell_trigger,
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_pin_bar,
)


def _make_df(closes, opens=None, highs=None, lows=None, volumes=None, n=None):
    n = n or len(closes)
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n),
        "open": opens or closes,
        "high": highs or [c * 1.01 for c in closes],
        "low": lows or [c * 0.99 for c in closes],
        "close": closes,
        "volume": volumes or [1000] * n,
    })


# ==============================================================================
# Test: nến đảo chiều
# ==============================================================================

class TestCandlePatterns:
    def test_bullish_engulfing_detected(self):
        df = _make_df(
            closes=[95, 105], opens=[100, 94],  # phiên 1 giảm (100->95), phiên 2 tăng nuốt trọn (94->105)
        )
        assert is_bullish_engulfing(df) is True

    def test_bearish_engulfing_detected(self):
        df = _make_df(
            closes=[105, 95], opens=[100, 106],
        )
        assert is_bearish_engulfing(df) is True

    def test_no_engulfing_for_small_candle(self):
        df = _make_df(closes=[100, 101], opens=[99, 100])
        assert is_bullish_engulfing(df) is False

    def test_bullish_pin_bar_detected(self):
        # Thân nhỏ ở trên, bóng dưới dài
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=1),
            "open": [100], "high": [100.6], "low": [90], "close": [100.5],
            "volume": [1000],
        })
        assert is_pin_bar(df, direction="bullish") is True

    def test_bearish_pin_bar_detected(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=1),
            "open": [100], "high": [110], "low": [99.2], "close": [99.5],
            "volume": [1000],
        })
        assert is_pin_bar(df, direction="bearish") is True

    def test_raises_for_invalid_direction(self):
        df = _make_df(closes=[100])
        with pytest.raises(InvalidStockSignalError):
            is_pin_bar(df, direction="sideways")


# ==============================================================================
# Test: check_base_trend_condition
# ==============================================================================

class TestCheckBaseTrendCondition:
    def test_passes_when_above_ema200_and_strong_adx(self):
        assert check_base_trend_condition(close=110, ema200=100, adx=30) is True

    def test_fails_when_below_ema200(self):
        assert check_base_trend_condition(close=90, ema200=100, adx=30) is False

    def test_fails_when_adx_weak(self):
        assert check_base_trend_condition(close=110, ema200=100, adx=15) is False

    def test_fails_when_data_missing(self):
        assert check_base_trend_condition(close=110, ema200=None, adx=30) is False


# ==============================================================================
# Test: check_pullback_pattern
# ==============================================================================

class TestCheckPullbackPattern:
    def test_detects_valid_pullback(self):
        # Giá hiện tại nằm trong vùng EMA20/50, có nến bullish engulfing, RSI hồi từ 40->45
        closes = [100] * 20 + [95, 101]
        opens = [100] * 20 + [100, 94]
        df = _make_df(closes, opens=opens)
        rsi_series = pd.Series([45.0] * 19 + [40.0, 45.0])
        result = check_pullback_pattern(df, ema20=100, ema50=101, rsi_series=rsi_series)
        assert result is True

    def test_fails_when_far_from_ema_zone(self):
        closes = [100] * 20 + [95, 130]  # giá cuối quá xa vùng EMA
        opens = [100] * 20 + [100, 94]
        df = _make_df(closes, opens=opens)
        rsi_series = pd.Series([45.0] * 19 + [40.0, 45.0])
        result = check_pullback_pattern(df, ema20=100, ema50=101, rsi_series=rsi_series)
        assert result is False


# ==============================================================================
# Test: check_breakout_pattern
# ==============================================================================

class TestCheckBreakoutPattern:
    def test_detects_valid_breakout(self):
        df = _make_df(closes=[100, 110], volumes=[1000, 2000])
        result = check_breakout_pattern(df, resistance_level=105, volume_ma20=1000, multiplier=1.5)
        assert result is True

    def test_fails_without_volume_confirmation(self):
        df = _make_df(closes=[100, 110], volumes=[1000, 1100])  # volume không đủ đột biến
        result = check_breakout_pattern(df, resistance_level=105, volume_ma20=1000, multiplier=1.5)
        assert result is False

    def test_fails_when_price_does_not_break_resistance(self):
        df = _make_df(closes=[100, 103], volumes=[1000, 2000])
        result = check_breakout_pattern(df, resistance_level=105, volume_ma20=1000, multiplier=1.5)
        assert result is False


# ==============================================================================
# Test: check_support_bounce_pattern
# ==============================================================================

class TestCheckSupportBouncePattern:
    def test_detects_valid_bounce(self):
        df = _make_df(closes=[100, 100.5])
        rsi_series = pd.Series([35.0, 38.0])
        result = check_support_bounce_pattern(df, support_level=100, rsi_series=rsi_series)
        assert result is True

    def test_fails_when_far_from_support(self):
        df = _make_df(closes=[100, 120])
        rsi_series = pd.Series([35.0, 38.0])
        result = check_support_bounce_pattern(df, support_level=100, rsi_series=rsi_series)
        assert result is False

    def test_fails_when_rsi_not_recovering(self):
        df = _make_df(closes=[100, 100.5])
        rsi_series = pd.Series([38.0, 35.0])  # RSI đang giảm, không phải hồi phục
        result = check_support_bounce_pattern(df, support_level=100, rsi_series=rsi_series)
        assert result is False


# ==============================================================================
# Test: check_stop_loss_hit
# ==============================================================================

class TestCheckStopLossHit:
    def test_triggers_on_stop_loss_price(self):
        assert check_stop_loss_hit(current_price=95, stop_loss_price=96) is True

    def test_triggers_on_recent_low_break(self):
        assert check_stop_loss_hit(current_price=95, recent_low=96) is True

    def test_triggers_on_excess_risk(self):
        assert check_stop_loss_hit(current_price=100, current_loss_pct_nav=0.025) is True

    def test_no_trigger_when_all_safe(self):
        assert check_stop_loss_hit(
            current_price=110, stop_loss_price=100, recent_low=105, current_loss_pct_nav=0.01,
        ) is False

    def test_no_trigger_when_no_data_provided(self):
        assert check_stop_loss_hit(current_price=100) is False


# ==============================================================================
# Test: check_resistance_overbought / check_volume_depletion / check_ema200_break_confirmed
# ==============================================================================

class TestSellTechnicalChecks:
    def test_resistance_overbought_detected(self):
        df = _make_df(closes=[100, 104])
        rsi_series = pd.Series([65.0, 75.0])
        assert check_resistance_overbought(df, resistance_level=105, rsi_series=rsi_series) is True

    def test_resistance_overbought_fails_when_rsi_not_high(self):
        df = _make_df(closes=[100, 104])
        rsi_series = pd.Series([65.0, 68.0])
        assert check_resistance_overbought(df, resistance_level=105, rsi_series=rsi_series) is False

    def test_volume_depletion_detected(self):
        closes = [100, 101, 102, 103, 104, 105]  # giá tăng đều
        volumes = [5000, 4500, 4000, 3500, 3000, 2500]  # khối lượng giảm dần
        df = _make_df(closes, volumes=volumes)
        assert check_volume_depletion(df, lookback=5) is True

    def test_volume_depletion_fails_when_volume_not_declining(self):
        closes = [100, 101, 102, 103, 104, 105]
        volumes = [1000, 1000, 1000, 1000, 1000, 1000]
        df = _make_df(closes, volumes=volumes)
        assert check_volume_depletion(df, lookback=5) is False

    def test_ema200_break_confirmed(self):
        df = _make_df(closes=[95, 94])
        ema200_series = pd.Series([100.0, 100.0])
        assert check_ema200_break_confirmed(df, ema200_series, adx=30, sessions=2) is True

    def test_ema200_break_not_confirmed_without_adx(self):
        df = _make_df(closes=[95, 94])
        ema200_series = pd.Series([100.0, 100.0])
        assert check_ema200_break_confirmed(df, ema200_series, adx=15, sessions=2) is False


# ==============================================================================
# Test: evaluate_fundamental_buy_screen / sell_screen
# ==============================================================================

class TestFundamentalScreens:
    def test_none_fundamentals_returns_neutral(self):
        result = evaluate_fundamental_buy_screen(None)
        assert result["dat"] is None

    def test_passes_when_all_criteria_met(self):
        fundamentals = {
            "eps_growth_yoy": 0.18, "eps_growth_quarters_streak": 3,
            "pe": 8.5, "pe_industry_avg": 11.0, "peg": 0.8, "roe": 0.17,
            "de": 0.5, "de_industry_avg": 0.7, "cfo": 100, "cfo_growth": 0.1,
        }
        result = evaluate_fundamental_buy_screen(fundamentals)
        assert result["dat"] is True

    def test_fails_when_roe_too_low(self):
        fundamentals = {
            "eps_growth_yoy": 0.18, "eps_growth_quarters_streak": 3,
            "pe": 8.5, "pe_industry_avg": 11.0, "peg": 0.8, "roe": 0.05,  # ROE quá thấp
            "de": 0.5, "de_industry_avg": 0.7, "cfo": 100, "cfo_growth": 0.1,
        }
        result = evaluate_fundamental_buy_screen(fundamentals)
        assert result["dat"] is False

    def test_sell_screen_none_returns_neutral(self):
        result = evaluate_fundamental_sell_screen(None)
        assert result["dat"] is None

    def test_sell_screen_detects_eps_decline(self):
        result = evaluate_fundamental_sell_screen({"eps_decline_quarters_streak": 3})
        assert result["dat"] is True
        assert any("EPS" in r for r in result["ly_do"])

    def test_sell_screen_no_signal_when_healthy(self):
        result = evaluate_fundamental_sell_screen({"eps_decline_quarters_streak": 0})
        assert result["dat"] is False


# ==============================================================================
# Test: check_buy_veto
# ==============================================================================

class TestCheckBuyVeto:
    def test_vetoes_on_severe_negative_macro_and_downtrend(self):
        result = check_buy_veto(macro_score=-1.5, market_regime="DOWNTREND")
        assert result["phu_quyet"] is True

    def test_no_veto_when_macro_negative_but_not_downtrend(self):
        result = check_buy_veto(macro_score=-1.5, market_regime="SIDEWAY")
        assert result["phu_quyet"] is False

    def test_vetoes_on_low_liquidity(self):
        result = check_buy_veto(
            macro_score=None, market_regime=None,
            avg_volume_20=1000, min_liquidity_threshold=50000,
        )
        assert result["phu_quyet"] is True

    def test_no_veto_when_all_clear(self):
        result = check_buy_veto(macro_score=0.5, market_regime="UPTREND")
        assert result["phu_quyet"] is False


# ==============================================================================
# Test: evaluate_stock_signal — luồng quyết định chính (5 bước)
# ==============================================================================

class TestEvaluateStockSignal:
    def _uptrend_df(self, n=260):
        closes = [100 + i * 0.3 for i in range(n)]
        return _make_df(closes, n=n)

    def test_stop_loss_takes_absolute_priority(self):
        df = self._uptrend_df()
        result = evaluate_stock_signal(
            "HPG", df, position_info={"gia_cat_lo": df["close"].iloc[-1] + 1},
        )
        assert result["khuyen_nghi"] == "BAN"
        assert result["loai_ban"] == "CAT_LO"
        assert result["uu_tien"] == "CAO"

    def test_fundamental_sell_signal_overrides_buy_technical(self):
        df = self._uptrend_df()
        result = evaluate_stock_signal(
            "HPG", df,
            fundamentals={"eps_decline_quarters_streak": 3},
        )
        assert result["khuyen_nghi"] == "BAN"
        assert result["loai_ban"] == "CHOT_LOI"
        assert result["uu_tien"] == "CAO"

    def test_veto_blocks_buy_even_with_good_technical(self):
        df = self._uptrend_df()
        result = evaluate_stock_signal(
            "HPG", df, macro_score=-1.5, market_regime="DOWNTREND",
        )
        assert result["khuyen_nghi"] != "MUA"
        assert len(result["canh_bao"]) > 0

    def test_raises_for_empty_df(self):
        with pytest.raises(InvalidStockSignalError):
            evaluate_stock_signal("HPG", pd.DataFrame())

    def test_output_structure_matches_spec(self):
        df = self._uptrend_df()
        result = evaluate_stock_signal("HPG", df)
        expected_keys = {
            "ma", "khuyen_nghi", "loai_ban", "uu_tien", "stock_score",
            "fundamental_score", "technical_score", "chi_tiet",
            "khoang_gia_vao_lenh_de_xuat", "canh_bao", "ghi_chu",
        }
        assert expected_keys.issubset(result.keys())
        assert result["khuyen_nghi"] in ("MUA", "GIU_THEO_DOI", "BAN")


# ==============================================================================
# Test: build_signal_summary_report
# ==============================================================================

class TestBuildSignalSummaryReport:
    def test_groups_by_recommendation(self):
        evaluations = [
            {"ma": "AAA", "khuyen_nghi": "MUA", "stock_score": 0.8},
            {"ma": "BBB", "khuyen_nghi": "MUA", "stock_score": 1.5},
            {"ma": "CCC", "khuyen_nghi": "BAN", "loai_ban": "CAT_LO"},
            {"ma": "DDD", "khuyen_nghi": "BAN", "loai_ban": "CHOT_LOI", "uu_tien": "CAO"},
            {"ma": "EEE", "khuyen_nghi": "GIU_THEO_DOI"},
        ]
        report = build_signal_summary_report(evaluations)

        assert report["tong_so_ma"] == 5
        assert [e["ma"] for e in report["mua"]] == ["BBB", "AAA"]  # sắp theo score giảm dần
        assert len(report["ban_cat_lo"]) == 1
        assert len(report["ban_chot_loi"]) == 1
        assert len(report["giu_theo_doi"]) == 1

    def test_empty_input(self):
        report = build_signal_summary_report([])
        assert report["tong_so_ma"] == 0
        assert report["mua"] == []
