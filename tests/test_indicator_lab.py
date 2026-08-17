"""Unit test cho experimental/indicator_lab.py — KHÔNG cần viết lại test
cho RSI/ATR/Bollinger/VolumeMA (đã có sẵn ở tests/test_indicators.py và
tests/test_market_breadth.py). Chỉ test phần MỚI: hàm tín hiệu, engine,
công thức lợi nhuận ròng, và quét watchlist."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experimental.indicator_lab import (
    BO_LOC_MAC_DINH,
    THAM_SO_MAC_DINH,
    TRAILING_TP_TIERS_MAC_DINH,
    InvalidIndicatorLabError,
    candle_body_pct,
    candle_range_pct,
    chay_backtest,
    danh_gia_tin_hieu,
    kiem_tra_bo_loc_bo_sung,
    kiem_tra_dieu_kien_buy_goc,
    kiem_tra_dieu_kien_sell_goc,
    quet_watchlist_tim_tin_hieu,
    tinh_loi_nhuan_rong,
    tinh_toan_chi_bao,
)


def _make_realistic_df(n=1000, seed=42, xu_huong=0.02):
    np.random.seed(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    closes = 50 + np.cumsum(np.random.normal(xu_huong, 1.0, n))
    opens = closes * (1 + np.random.normal(0, 0.005, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.01, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.01, n)))
    volumes = np.random.randint(500_000, 2_000_000, n).astype(float)
    return pd.DataFrame({
        "date": dates, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


class TestCandleFormulas:
    def test_candle_body_pct(self):
        df = pd.DataFrame({"high": [110.0], "low": [100.0]})
        assert candle_body_pct(df, 0) == pytest.approx(10.0)

    def test_candle_body_pct_none_when_low_zero(self):
        df = pd.DataFrame({"high": [10.0], "low": [0.0]})
        assert candle_body_pct(df, 0) is None

    def test_candle_range_pct(self):
        df = pd.DataFrame({
            "high": [100.0, 105.0, 103.0, 110.0],
            "low": [95.0, 98.0, 99.0, 102.0],
        })
        # 3 nến liền trước nến index 3 (index 0,1,2): high max=105, low min=95
        result = candle_range_pct(df, 3, 3)
        assert result == pytest.approx((105.0 - 95.0) / 95.0 * 100)

    def test_candle_range_pct_none_when_not_enough_history(self):
        df = pd.DataFrame({"high": [100.0], "low": [95.0]})
        assert candle_range_pct(df, 0, 3) is None


class TestKiemTraDieuKienGoc:
    def _make_setup_df(self):
        # 5 nến nền đi ngang hẹp, nến thứ 6 breakout mạnh lên trên.
        highs = [100, 101, 100.5, 101, 100.8, 108]
        lows = [98, 99, 98.5, 99.2, 98.8, 101]
        closes = [99, 100, 99.8, 100.2, 99.9, 107]
        opens = [98.5, 99.5, 100, 99.9, 100, 101.5]
        n = len(highs)
        return pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [1_000_000.0] * n,
        })

    def test_buy_returns_none_when_not_enough_history(self):
        df = self._make_setup_df()
        tham_so = {**THAM_SO_MAC_DINH, "buy_lookback": 4, "ema_period": 3, "ma_period": 2}
        chi_bao = tinh_toan_chi_bao(df, tham_so, BO_LOC_MAC_DINH)
        assert kiem_tra_dieu_kien_buy_goc(df, 2, tham_so, chi_bao) is None

    def test_buy_detects_breakout_with_loose_range(self):
        df = self._make_setup_df()
        tham_so = {
            "buy_lookback": 4, "sell_lookback": 4, "ema_period": 3, "ma_period": 2,
            "range_pct_max": 10.0, "body_pct_min": 0.1, "body_pct_max": 20.0,
        }
        chi_bao = tinh_toan_chi_bao(df, tham_so, BO_LOC_MAC_DINH)
        ket_qua = kiem_tra_dieu_kien_buy_goc(df, 5, tham_so, chi_bao)
        assert ket_qua is True

    def test_buy_false_when_body_pct_out_of_range(self):
        df = self._make_setup_df()
        tham_so = {
            "buy_lookback": 4, "sell_lookback": 4, "ema_period": 3, "ma_period": 2,
            "range_pct_max": 10.0, "body_pct_min": 0.1, "body_pct_max": 1.0,  # quá chặt -> loại vì "trap"
        }
        chi_bao = tinh_toan_chi_bao(df, tham_so, BO_LOC_MAC_DINH)
        assert kiem_tra_dieu_kien_buy_goc(df, 5, tham_so, chi_bao) is False

    def test_sell_symmetric_to_buy(self):
        # Đảo ngược dữ liệu breakout thành breakdown.
        highs = [102, 101, 101.5, 100.8, 101.2, 99]
        lows = [100, 99, 99.5, 98.8, 99.2, 92]
        closes = [101, 100, 100.2, 99.8, 100.1, 93]
        opens = [101.5, 100.5, 100, 100.1, 100, 99.5]
        n = len(highs)
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [1_000_000.0] * n,
        })
        tham_so = {
            "buy_lookback": 4, "sell_lookback": 4, "ema_period": 3, "ma_period": 2,
            "range_pct_max": 10.0, "body_pct_min": 0.1, "body_pct_max": 20.0,
        }
        chi_bao = tinh_toan_chi_bao(df, tham_so, BO_LOC_MAC_DINH)
        assert kiem_tra_dieu_kien_sell_goc(df, 5, tham_so, chi_bao) is True


class TestBoLocBoSung:
    def _chi_bao_gia_lap(self, gia_tri):
        return {k: pd.Series([gia_tri]) for k in ("rsi", "atr", "bb_upper", "bb_lower")} | {"atr_median": 5.0}

    def test_rsi_filter_chan_buy_khi_duoi_50(self):
        df = pd.DataFrame({"close": [100.0], "volume": [1000.0]})
        chi_bao = self._chi_bao_gia_lap(30.0)
        chi_bao["rsi"] = pd.Series([30.0])  # < 50
        bo_loc = {**BO_LOC_MAC_DINH, "rsi_enabled": True}
        assert kiem_tra_bo_loc_bo_sung(df, 0, chi_bao, bo_loc, "BUY") is False

    def test_rsi_filter_cho_qua_buy_khi_tren_50(self):
        df = pd.DataFrame({"close": [100.0], "volume": [1000.0]})
        chi_bao = self._chi_bao_gia_lap(60.0)
        chi_bao["rsi"] = pd.Series([60.0])
        bo_loc = {**BO_LOC_MAC_DINH, "rsi_enabled": True}
        assert kiem_tra_bo_loc_bo_sung(df, 0, chi_bao, bo_loc, "BUY") is True

    def test_volume_filter(self):
        df = pd.DataFrame({"close": [100.0], "volume": [500.0]})
        chi_bao = {"volume_ma": pd.Series([1000.0])}
        bo_loc = {**BO_LOC_MAC_DINH, "volume_enabled": True}
        assert kiem_tra_bo_loc_bo_sung(df, 0, chi_bao, bo_loc, "BUY") is False  # 500 < 1000

        chi_bao2 = {"volume_ma": pd.Series([100.0])}
        assert kiem_tra_bo_loc_bo_sung(df, 0, chi_bao2, bo_loc, "BUY") is True  # 500 > 100

    def test_atr_filter(self):
        df = pd.DataFrame({"close": [100.0], "volume": [1000.0]})
        chi_bao_thap = {"atr": pd.Series([2.0]), "atr_median": 5.0}
        bo_loc = {**BO_LOC_MAC_DINH, "atr_enabled": True}
        assert kiem_tra_bo_loc_bo_sung(df, 0, chi_bao_thap, bo_loc, "BUY") is False

        chi_bao_cao = {"atr": pd.Series([8.0]), "atr_median": 5.0}
        assert kiem_tra_bo_loc_bo_sung(df, 0, chi_bao_cao, bo_loc, "BUY") is True

    def test_bollinger_filter_buy_can_vuot_dai_tren(self):
        df = pd.DataFrame({"close": [100.0], "volume": [1000.0]})
        chi_bao = {"bb_upper": pd.Series([105.0])}
        bo_loc = {**BO_LOC_MAC_DINH, "bollinger_enabled": True}
        assert kiem_tra_bo_loc_bo_sung(df, 0, chi_bao, bo_loc, "BUY") is False  # 100 < 105

        chi_bao2 = {"bb_upper": pd.Series([95.0])}
        assert kiem_tra_bo_loc_bo_sung(df, 0, chi_bao2, bo_loc, "BUY") is True  # 100 > 95

    def test_bollinger_filter_sell_can_duoi_dai_duoi(self):
        df = pd.DataFrame({"close": [100.0], "volume": [1000.0]})
        chi_bao = {"bb_lower": pd.Series([95.0])}
        bo_loc = {**BO_LOC_MAC_DINH, "bollinger_enabled": True}
        assert kiem_tra_bo_loc_bo_sung(df, 0, chi_bao, bo_loc, "SELL") is False  # 100 > 95

        chi_bao2 = {"bb_lower": pd.Series([105.0])}
        assert kiem_tra_bo_loc_bo_sung(df, 0, chi_bao2, bo_loc, "SELL") is True  # 100 < 105

    def test_khong_bat_filter_nao_luon_cho_qua(self):
        df = pd.DataFrame({"close": [100.0], "volume": [1000.0]})
        assert kiem_tra_bo_loc_bo_sung(df, 0, {}, BO_LOC_MAC_DINH, "BUY") is True


class TestChayBacktest:
    def test_engine_chay_khong_loi_va_co_cau_truc_dung(self):
        df = _make_realistic_df()
        kq = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        assert "trades" in kq and "open_position" in kq
        assert kq["so_lenh_da_dong"] == len(kq["trades"])
        assert kq["so_lan_thang"] + kq["so_lan_thua"] == kq["so_lenh_da_dong"]

    def test_open_position_khong_bi_lan_vao_trades(self):
        # Cắt dữ liệu ngay sau 1 điểm entry đã biết trước (không đủ phiên
        # để chạm SL/TP) -> phải ra open_position, KHÔNG có trong trades.
        df = _make_realistic_df()
        kq_full = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        assert kq_full["trades"], "Cần ít nhất 1 lệnh trong dữ liệu test để dựng kịch bản"
        entry_date_dau_tien = kq_full["trades"][0]["entry_date"]

        df_cat = df[df["date"].astype(str) <= entry_date_dau_tien].reset_index(drop=True)
        kq_cat = chay_backtest(df_cat, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        assert kq_cat["open_position"] is not None
        assert kq_cat["so_lenh_da_dong"] == 0
        assert kq_cat["open_position"]["entry_date"] == entry_date_dau_tien

    def test_open_position_cong_thuc_pnl_tam_tinh_dung(self):
        df = _make_realistic_df()
        kq_full = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        entry_date_dau_tien = kq_full["trades"][0]["entry_date"]
        entry_price_dau_tien = kq_full["trades"][0]["entry_price"]

        df_cat = df[df["date"].astype(str) <= entry_date_dau_tien].reset_index(drop=True)
        kq_cat = chay_backtest(df_cat, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        op = kq_cat["open_position"]
        gia_cuoi = float(df_cat["close"].iloc[-1])
        pnl_ky_vong = round((gia_cuoi - entry_price_dau_tien) / entry_price_dau_tien * 100, 2) if op["side"] == "LONG" else round((entry_price_dau_tien - gia_cuoi) / entry_price_dau_tien * 100, 2)
        assert op["unrealized_pnl_pct"] == pytest.approx(pnl_ky_vong, abs=0.05)

    def test_raises_for_invalid_von_ban_dau(self):
        df = _make_realistic_df(n=300)
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH, von_ban_dau=0)

    def test_raises_for_invalid_ty_trong_von(self):
        df = _make_realistic_df(n=300)
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH, ty_trong_von_pct=150)

    def test_raises_for_invalid_tp_tiers_khong_tang_dan(self):
        df = _make_realistic_df(n=300)
        tiers_sai = [{"muc_lai_pct": 10.0, "chot_pct_khoi_luong": 50}, {"muc_lai_pct": 5.0, "chot_pct_khoi_luong": 50}]
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, tiers_sai)

    def test_raises_for_tp_tiers_vuot_100_phan_tram(self):
        df = _make_realistic_df(n=300)
        tiers_sai = [{"muc_lai_pct": 5.0, "chot_pct_khoi_luong": 60}, {"muc_lai_pct": 10.0, "chot_pct_khoi_luong": 60}]
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, tiers_sai)

    def test_them_bo_loc_lam_giam_so_luong_tin_hieu(self):
        df = _make_realistic_df()
        kq_khong_loc = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        bo_loc_chat = {**BO_LOC_MAC_DINH, "rsi_enabled": True, "volume_enabled": True}
        kq_co_loc = chay_backtest(df, THAM_SO_MAC_DINH, bo_loc_chat, TRAILING_TP_TIERS_MAC_DINH)
        tong_lenh_khong_loc = kq_khong_loc["so_lenh_da_dong"] + (1 if kq_khong_loc["open_position"] else 0)
        tong_lenh_co_loc = kq_co_loc["so_lenh_da_dong"] + (1 if kq_co_loc["open_position"] else 0)
        assert tong_lenh_co_loc <= tong_lenh_khong_loc


    def test_chi_giao_dich_mot_chieu_long_bo_qua_short(self):
        df = _make_realistic_df()
        kq_ca_2_chieu = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        kq_chi_long = chay_backtest(
            df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
            chi_giao_dich_mot_chieu="LONG",
        )
        assert all(t["side"] == "LONG" for t in kq_chi_long["trades"])
        if kq_chi_long["open_position"]:
            assert kq_chi_long["open_position"]["side"] == "LONG"
        tong_ca_2 = kq_ca_2_chieu["so_lenh_da_dong"] + (1 if kq_ca_2_chieu["open_position"] else 0)
        tong_chi_long = kq_chi_long["so_lenh_da_dong"] + (1 if kq_chi_long["open_position"] else 0)
        assert tong_chi_long <= tong_ca_2

    def test_raises_for_invalid_chi_giao_dich_mot_chieu(self):
        df = _make_realistic_df(n=300)
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest(
                df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
                chi_giao_dich_mot_chieu="NGANG",
            )


class TestTinhLoiNhuanRong:
    def test_khong_co_lenh_nao_loi_nhuan_bang_0(self):
        kq = tinh_loi_nhuan_rong([], 1_000_000_000, 50.0)
        assert kq["loi_nhuan_rong"] == 0
        assert kq["von_cuoi_cung"] == 1_000_000_000

    def test_compound_dung_thu_tu(self):
        trades = [{"final_pnl_pct": 10.0}, {"final_pnl_pct": -5.0}]
        von_ban_dau = 1_000_000_000
        ty_trong = 50.0
        kq = tinh_loi_nhuan_rong(trades, von_ban_dau, ty_trong)

        equity = von_ban_dau
        equity += equity * 0.5 * 0.10
        equity += equity * 0.5 * (-0.05)
        assert kq["von_cuoi_cung"] == round(equity)


class TestQuetWatchlist:
    def test_quet_phat_hien_dung_tin_hieu_tai_nen_cuoi(self):
        df = _make_realistic_df()
        kq_full = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        entry_date_dau_tien = kq_full["trades"][0]["entry_date"]
        df_cat = df[df["date"].astype(str) <= entry_date_dau_tien].reset_index(drop=True)

        kq_quet = quet_watchlist_tim_tin_hieu({"MA_TEST": df_cat}, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH)
        tong_ma_phat_hien = len(kq_quet["buy"]) + len(kq_quet["sell"])
        assert tong_ma_phat_hien == 1

    def test_quet_bo_qua_ma_khong_du_du_lieu(self):
        df_ngan = _make_realistic_df(n=50)
        kq = quet_watchlist_tim_tin_hieu({"MA_NGAN": df_ngan}, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH)
        assert kq["buy"] == [] and kq["sell"] == []

    def test_quet_them_bo_loc_lam_giam_so_ma_lot_qua(self):
        df1 = _make_realistic_df(seed=1)
        df2 = _make_realistic_df(seed=2)
        df3 = _make_realistic_df(seed=3)
        du_lieu = {"A": df1, "B": df2, "C": df3}

        kq_khong_loc = quet_watchlist_tim_tin_hieu(du_lieu, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH)
        bo_loc_chat = {**BO_LOC_MAC_DINH, "rsi_enabled": True, "bollinger_enabled": True}
        kq_co_loc = quet_watchlist_tim_tin_hieu(du_lieu, THAM_SO_MAC_DINH, bo_loc_chat)

        tong_khong_loc = len(kq_khong_loc["buy"]) + len(kq_khong_loc["sell"])
        tong_co_loc = len(kq_co_loc["buy"]) + len(kq_co_loc["sell"])
        assert tong_co_loc <= tong_khong_loc

    def test_khong_loi_khi_watchlist_rong(self):
        kq = quet_watchlist_tim_tin_hieu({}, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH)
        assert kq == {"buy": [], "sell": []}

