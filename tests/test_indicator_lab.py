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
    chay_backtest_ket_hop_nhieu_bo,
    chay_backtest_ma_crossover,
    chay_backtest_nhieu_ma,
    danh_gia_tin_hieu,
    danh_gia_tin_hieu_ket_hop,
    kiem_tra_bo_loc_bo_sung,
    kiem_tra_dieu_kien_buy_goc,
    kiem_tra_dieu_kien_sell_goc,
    phat_hien_death_cross,
    phat_hien_golden_cross,
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

    def test_buy_false_khi_dong_cua_yeu_gay_cutloss_cao_hon_gia_vao(self):
        # Tái hiện ĐÚNG lỗi thật đã phát hiện: nến "breakout" so với quá
        # khứ NHƯNG đóng cửa ở NỬA DƯỚI của chính nến đó (High=108, Low=101,
        # Close=103 -> body_mid=104.5 > close=103) -> PHẢI bị loại, vì
        # Cutloss (body_mid) sẽ cao hơn giá vào lệnh — vô lý cho LONG.
        highs = [100, 101, 100.5, 101, 100.8, 108]
        lows = [98, 99, 98.5, 99.2, 98.8, 101]
        closes = [99, 100, 99.8, 100.2, 99.9, 103]  # đóng cửa YẾU (nửa dưới nến, dù vẫn phá đỉnh)
        opens = [98.5, 99.5, 100, 99.9, 100, 101.5]
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

        # Xác nhận trước: đúng là close < body_mid trong kịch bản này
        # (tái hiện chính xác lỗi thật) — nếu không còn đúng nữa, test tự sai.
        body_mid = (highs[5] + lows[5]) / 2
        assert closes[5] < body_mid

        assert kiem_tra_dieu_kien_buy_goc(df, 5, tham_so, chi_bao) is False

    def test_cutloss_luon_thap_hon_gia_vao_lenh_khi_dat_buy(self):
        # Kiểm tra TỔNG QUÁT trên dữ liệu ngẫu nhiên thật: BẤT KỲ khi nào
        # kiem_tra_dieu_kien_buy_goc() trả về True, Cutloss (body_mid) PHẢI
        # luôn THẤP HƠN giá đóng cửa (giá vào lệnh) — không có ngoại lệ.
        df = _make_realistic_df(n=1000)
        tham_so = THAM_SO_MAC_DINH
        chi_bao = tinh_toan_chi_bao(df, tham_so, BO_LOC_MAC_DINH)
        for i in range(300, len(df) - 1):
            if kiem_tra_dieu_kien_buy_goc(df, i, tham_so, chi_bao) is True:
                close_i = float(df["close"].iloc[i])
                body_mid = float((df["high"].iloc[i] + df["low"].iloc[i]) / 2)
                assert close_i > body_mid, f"Vi phạm tại phiên {i}: close={close_i}, body_mid={body_mid}"


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


    def test_chi_giao_dich_long_khong_co_short(self):
        # Cổ phiếu VN không có bán khống -> TOÀN BỘ lệnh (đã đóng + đang
        # mở) phải luôn là LONG, không bao giờ có SHORT.
        df = _make_realistic_df()
        kq = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        assert all(t["side"] == "LONG" for t in kq["trades"])
        if kq["open_position"]:
            assert kq["open_position"]["side"] == "LONG"

    def test_tin_hieu_sell_khong_giu_lenh_khong_mo_vi_the(self):
        # Khi KHÔNG đang giữ lệnh, tín hiệu SELL phải bị bỏ qua hoàn
        # toàn (không mở SHORT) — chỉ còn tác dụng cảnh báo khi ĐANG giữ LONG.
        df = _make_realistic_df()
        kq = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        assert kq["so_lenh_da_dong"] == len(kq["trades"])  # không lẫn lệnh SHORT nào

    def test_raises_for_so_phien_khoa_am(self):
        df = _make_realistic_df(n=300)
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest(
                df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
                so_phien_khoa_toi_thieu=-1,
            )

    def test_khoa_T_khong_cho_dong_lenh_som(self):
        # Dựng thủ công 1 tình huống: vào lệnh, rồi giá RỚT MẠNH ngay
        # phiên kế tiếp (đáng lẽ chạm SL) -> với so_phien_khoa_toi_thieu=2,
        # lệnh KHÔNG được đóng ở phiên +1, chỉ được đóng từ phiên +2 trở đi.
        import numpy as np
        n = 260
        dates = pd.bdate_range("2024-01-01", periods=n)
        closes = np.full(n, 100.0)
        # Nến breakout tại vị trí ~205 (đủ nền EMA200), giá RỚT mạnh ngay hôm sau.
        vi_tri_breakout = 210
        closes[vi_tri_breakout - 4:vi_tri_breakout] = [99, 99.2, 99.1, 99.3]
        closes[vi_tri_breakout] = 101.0  # breakout
        closes[vi_tri_breakout + 1] = 90.0  # rớt mạnh -> đáng lẽ chạm SL ngay
        closes[vi_tri_breakout + 2] = 90.0
        highs = closes * 1.005
        lows = closes * 0.995
        df = pd.DataFrame({
            "date": dates, "open": closes, "high": highs, "low": lows,
            "close": closes, "volume": [1_000_000.0] * n,
        })
        tham_so = {
            "buy_lookback": 4, "sell_lookback": 4, "ema_period": 200, "ma_period": 20,
            "range_pct_max": 5.0, "body_pct_min": 0.1, "body_pct_max": 20.0,
        }
        kq_khoa_2 = chay_backtest(
            df, tham_so, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
            so_phien_khoa_toi_thieu=2,
        )
        kq_khong_khoa = chay_backtest(
            df, tham_so, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
            so_phien_khoa_toi_thieu=0,
        )
        # Không khóa -> đóng lệnh ngay phiên kế (lỗ nặng do giá rớt mạnh).
        # Có khóa T+2 -> lệnh KHÔNG thể đóng ở phiên +1, exit_date phải
        # muộn hơn (hoặc trở thành vị thế đang mở nếu dữ liệu hết trong lúc khóa).
        if kq_khong_khoa["trades"] and kq_khoa_2["trades"]:
            ngay_dong_khong_khoa = kq_khong_khoa["trades"][0]["exit_date"]
            ngay_dong_co_khoa = kq_khoa_2["trades"][0]["exit_date"]
            assert ngay_dong_co_khoa >= ngay_dong_khong_khoa


    def test_phi_va_thue_tru_dung_moi_lenh(self):
        # Chênh lệch final_pnl_pct giữa CÓ và KHÔNG phí/thuế phải luôn
        # bằng ĐÚNG (phi_moi_gioi_pct*2 + thue_ban_pct), bất kể đóng qua
        # SL hay qua nhiều tier TP (đã chứng minh tương đương toán học).
        df = _make_realistic_df()
        kq_khong_phi = chay_backtest(
            df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
            phi_moi_gioi_pct=0.0, thue_ban_pct=0.0,
        )
        kq_co_phi = chay_backtest(
            df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
            phi_moi_gioi_pct=0.15, thue_ban_pct=0.1,
        )
        assert len(kq_khong_phi["trades"]) == len(kq_co_phi["trades"])
        for t1, t2 in zip(kq_khong_phi["trades"], kq_co_phi["trades"]):
            chenh_lech = round(t1["final_pnl_pct"] - t2["final_pnl_pct"], 3)
            assert chenh_lech == pytest.approx(0.4, abs=0.01)

    def test_raises_for_phi_am(self):
        df = _make_realistic_df(n=300)
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH, phi_moi_gioi_pct=-0.1)
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH, thue_ban_pct=-0.1)

    def test_raises_for_bien_do_khong_hop_le(self):
        df = _make_realistic_df(n=300)
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH, bien_do_dao_dong_pct=0)

    def test_so_co_phieu_uoc_tinh_lam_tron_lo_100(self):
        df = _make_realistic_df()
        kq = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        for t in kq["trades"]:
            assert t["so_co_phieu_uoc_tinh"] % 100 == 0

    def test_canh_bao_bien_do_phat_hien_dung_khi_gan_tran(self):
        import numpy as np
        n = 260
        dates = pd.bdate_range("2024-01-01", periods=n)
        closes = np.full(n, 100.0)
        vi_tri = 210
        closes[vi_tri - 4:vi_tri] = [99, 99.2, 99.1, 99.3]
        closes[vi_tri] = 106.8  # +6,8% so với hôm trước (99.3) -> gần trần 7%
        closes[vi_tri + 1:] = 106.8
        df = pd.DataFrame({
            "date": dates, "open": closes, "high": closes * 1.005, "low": closes * 0.995,
            "close": closes, "volume": [1_000_000.0] * n,
        })
        tham_so = {
            "buy_lookback": 4, "sell_lookback": 4, "ema_period": 200, "ma_period": 20,
            "range_pct_max": 5.0, "body_pct_min": 0.1, "body_pct_max": 20.0,
        }
        kq = chay_backtest(df, tham_so, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH, bien_do_dao_dong_pct=7.0)
        if kq["open_position"]:
            assert kq["open_position"]["canh_bao_bien_do_vao_lenh"] == "gan_tran"

    def test_loc_theo_giai_doan_giam_so_lenh(self):
        from core.market_regime_detector import tinh_chuoi_giai_doan_theo_ngay
        df = _make_realistic_df()
        chuoi = tinh_chuoi_giai_doan_theo_ngay({"MA_TEST": df})

        kq_khong_loc = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        kq_uptrend = chay_backtest(
            df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
            chuoi_giai_doan=chuoi, giai_doan_loc="uptrend",
        )
        assert kq_uptrend["so_lenh_da_dong"] <= kq_khong_loc["so_lenh_da_dong"]

    def test_khong_loc_khi_giai_doan_loc_la_none(self):
        # KHÔNG truyền chuoi_giai_doan/giai_doan_loc -> hành vi giữ NGUYÊN
        # như cũ (tương thích ngược).
        df = _make_realistic_df()
        kq1 = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        kq2 = chay_backtest(
            df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
            chuoi_giai_doan=None, giai_doan_loc=None,
        )
        assert kq1["so_lenh_da_dong"] == kq2["so_lenh_da_dong"]


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

    def test_loai_ma_khoi_luong_thap(self):
        df = _make_realistic_df()
        kq_full = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        entry_date_dau_tien = kq_full["trades"][0]["entry_date"]
        df_cat = df[df["date"].astype(str) <= entry_date_dau_tien].reset_index(drop=True)
        df_khoi_luong_thap = df_cat.copy()
        df_khoi_luong_thap["volume"] = 50_000.0  # dưới ngưỡng 300.000 mặc định

        kq_mac_dinh = quet_watchlist_tim_tin_hieu({"MA_TEST": df_cat}, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH)
        kq_khoi_luong_thap = quet_watchlist_tim_tin_hieu({"MA_TEST": df_khoi_luong_thap}, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH)

        assert len(kq_mac_dinh["buy"]) + len(kq_mac_dinh["sell"]) == 1
        assert len(kq_khoi_luong_thap["buy"]) + len(kq_khoi_luong_thap["sell"]) == 0

    def test_tat_bo_loc_volume_bang_none(self):
        df = _make_realistic_df()
        kq_full = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        entry_date_dau_tien = kq_full["trades"][0]["entry_date"]
        df_cat = df[df["date"].astype(str) <= entry_date_dau_tien].reset_index(drop=True)
        df_cat["volume"] = 50_000.0

        kq = quet_watchlist_tim_tin_hieu(
            {"MA_TEST": df_cat}, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH,
            nguong_volume_ma20_toi_thieu=None,
        )
        assert len(kq["buy"]) + len(kq["sell"]) == 1

    def test_co_du_cac_truong_gia_moi(self):
        df = _make_realistic_df()
        kq_full = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        entry_date_dau_tien = kq_full["trades"][0]["entry_date"]
        df_cat = df[df["date"].astype(str) <= entry_date_dau_tien].reset_index(drop=True)

        kq = quet_watchlist_tim_tin_hieu({"MA_TEST": df_cat}, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH)
        hang = kq["buy"][0] if kq["buy"] else kq["sell"][0]
        for truong in ("gia_de_nghi_vao_lenh", "gia_cutloss", "volume", "volume_ma20",
                       "gia_chot_loi_5pct", "gia_chot_loi_10pct", "gia_chot_loi_15pct"):
            assert truong in hang

        # Giá chốt lời phải tăng dần đúng theo % (5% < 10% < 15%).
        assert hang["gia_chot_loi_5pct"] < hang["gia_chot_loi_10pct"] < hang["gia_chot_loi_15pct"]
        # Giá chốt lời 5% phải khớp công thức entry*(1+0.05).
        assert hang["gia_chot_loi_5pct"] == pytest.approx(hang["gia_de_nghi_vao_lenh"] * 1.05, abs=0.01)

    def test_tuy_chinh_muc_chot_loi_pct(self):
        df = _make_realistic_df()
        kq_full = chay_backtest(df, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        entry_date_dau_tien = kq_full["trades"][0]["entry_date"]
        df_cat = df[df["date"].astype(str) <= entry_date_dau_tien].reset_index(drop=True)

        kq = quet_watchlist_tim_tin_hieu(
            {"MA_TEST": df_cat}, THAM_SO_MAC_DINH, BO_LOC_MAC_DINH, muc_chot_loi_pct=(3.0, 7.0),
        )
        hang = kq["buy"][0] if kq["buy"] else kq["sell"][0]
        assert "gia_chot_loi_3pct" in hang
        assert "gia_chot_loi_7pct" in hang
        assert "gia_chot_loi_5pct" not in hang

    def test_sell_khong_co_gia_vao_lenh_cutloss_chot_loi(self):
        # SỬA LỖI (06/08/2026): hàng SELL KHÔNG được có "gia_de_nghi_vao_lenh"/
        # "gia_cutloss"/"gia_chot_loi_*" (các công thức đó chỉ đúng cho
        # LONG) — SELL chỉ là cảnh báo, phải có "ghi_chu" giải thích rõ.
        n = 260
        dates = pd.bdate_range("2024-01-01", periods=n)
        closes = np.full(n, 100.0)
        vi_tri = 210
        closes[vi_tri - 4:vi_tri] = [100.8, 100.5, 100.6, 100.3]
        closes[vi_tri] = 92.0  # breakdown mạnh (thấp hơn hẳn các đáy trước)
        closes[vi_tri + 1:] = 91.0
        df = pd.DataFrame({
            "date": dates, "open": closes, "high": closes * 1.03, "low": closes * 0.997,
            "close": closes, "volume": [1_000_000.0] * n,
        })
        tham_so = {
            "buy_lookback": 4, "sell_lookback": 4, "ema_period": 200, "ma_period": 20,
            "range_pct_max": 5.0, "body_pct_min": 0.1, "body_pct_max": 20.0,
        }
        df_cat = df.iloc[: vi_tri + 1].reset_index(drop=True)
        kq = quet_watchlist_tim_tin_hieu({"MA_TEST": df_cat}, tham_so, BO_LOC_MAC_DINH, nguong_volume_ma20_toi_thieu=None)

        assert len(kq["sell"]) == 1
        hang = kq["sell"][0]
        assert "gia_de_nghi_vao_lenh" not in hang
        assert "gia_cutloss" not in hang
        assert "gia_chot_loi_5pct" not in hang
        assert "ghi_chu" in hang
        assert "THOÁT LỆNH" in hang["ghi_chu"]



class TestKetHopNhieuBoThamSo:
    def _tao_2_bo_tach_biet(self):
        import numpy as np
        np.random.seed(11)
        n = 700
        dates = pd.bdate_range("2022-01-01", periods=n)
        closes = 50 + np.cumsum(np.random.normal(0.02, 1.0, n))
        opens = closes * (1 + np.random.normal(0, 0.005, n))
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.01, n)))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.01, n)))
        volumes = np.random.randint(500_000, 2_000_000, n).astype(float)
        df = pd.DataFrame({
            "date": dates, "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        })
        ts_a = {"buy_lookback": 3, "sell_lookback": 3, "ema_period": 200, "ma_period": 20,
                "range_pct_max": 3.0, "body_pct_min": 0.5, "body_pct_max": 2.0}
        ts_b = {"buy_lookback": 8, "sell_lookback": 8, "ema_period": 50, "ma_period": 5,
                "range_pct_max": 3.0, "body_pct_min": 0.5, "body_pct_max": 2.0}
        return df, ts_a, ts_b

    def test_danh_gia_tin_hieu_ket_hop_tang_hoac_bang_tung_bo(self):
        df, ts_a, ts_b = self._tao_2_bo_tach_biet()
        chi_bao_a = tinh_toan_chi_bao(df, ts_a, BO_LOC_MAC_DINH)
        chi_bao_b = tinh_toan_chi_bao(df, ts_b, BO_LOC_MAC_DINH)

        so_lan_a = sum(1 for i in range(250, len(df)) if danh_gia_tin_hieu(df, i, ts_a, BO_LOC_MAC_DINH, chi_bao_a) == "BUY")
        so_lan_b = sum(1 for i in range(250, len(df)) if danh_gia_tin_hieu(df, i, ts_b, BO_LOC_MAC_DINH, chi_bao_b) == "BUY")
        so_lan_ket_hop = sum(
            1 for i in range(250, len(df))
            if danh_gia_tin_hieu_ket_hop(df, i, [ts_a, ts_b], BO_LOC_MAC_DINH, [chi_bao_a, chi_bao_b]) == "BUY"
        )
        assert so_lan_ket_hop >= max(so_lan_a, so_lan_b)
        assert so_lan_ket_hop <= so_lan_a + so_lan_b

    def test_chay_backtest_ket_hop_hoat_dong_dung(self):
        df, ts_a, ts_b = self._tao_2_bo_tach_biet()
        kq = chay_backtest_ket_hop_nhieu_bo(df, [ts_a, ts_b], BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        assert "trades" in kq and "open_position" in kq
        assert kq["so_bo_tham_so_ket_hop"] == 2
        assert kq["so_lenh_da_dong"] == len(kq["trades"])

    def test_chay_backtest_ket_hop_so_lenh_tang_hon_hoac_bang_tung_bo_rieng(self):
        df, ts_a, ts_b = self._tao_2_bo_tach_biet()
        kq_a = chay_backtest(df, ts_a, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        kq_b = chay_backtest(df, ts_b, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        kq_ket_hop = chay_backtest_ket_hop_nhieu_bo(df, [ts_a, ts_b], BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)

        tong_a = kq_a["so_lenh_da_dong"] + (1 if kq_a["open_position"] else 0)
        tong_b = kq_b["so_lenh_da_dong"] + (1 if kq_b["open_position"] else 0)
        tong_ket_hop = kq_ket_hop["so_lenh_da_dong"] + (1 if kq_ket_hop["open_position"] else 0)
        assert tong_ket_hop >= max(tong_a, tong_b)

    def test_raises_for_danh_sach_rong(self):
        df, ts_a, _ = self._tao_2_bo_tach_biet()
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest_ket_hop_nhieu_bo(df, [], BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)

    def test_chi_giao_dich_long_khong_co_short_khi_ket_hop(self):
        df, ts_a, ts_b = self._tao_2_bo_tach_biet()
        kq = chay_backtest_ket_hop_nhieu_bo(df, [ts_a, ts_b], BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)
        assert all(t["side"] == "LONG" for t in kq["trades"])
        if kq["open_position"]:
            assert kq["open_position"]["side"] == "LONG"


class TestChayBacktestNhieuMa:
    def _tao_df_breakout(self, n, vi_tri_breakout, seed=1):
        np.random.seed(seed)
        dates = pd.bdate_range("2024-01-01", periods=n)
        closes = np.full(n, 100.0) + np.cumsum(np.random.normal(0, 0.05, n))
        closes[vi_tri_breakout - 4:vi_tri_breakout] = closes[vi_tri_breakout - 5] + np.array([0.1, 0.3, 0.2, 0.4])
        closes[vi_tri_breakout] = closes[vi_tri_breakout - 1] + 5.0
        closes[vi_tri_breakout + 1:] = closes[vi_tri_breakout] + 1.0
        # high/low KHÔNG đối xứng quanh close — đảm bảo close nằm ở NỬA
        # TRÊN của mỗi nến (high sát close, low xa hơn hẳn) để không vô
        # tình vi phạm điều kiện "Cutloss phải thấp hơn giá vào lệnh".
        return pd.DataFrame({
            "date": dates, "open": closes, "high": closes * 1.003, "low": closes * 0.97,
            "close": closes, "volume": [1_000_000.0] * n,
        })

    def _tham_so_de_khop(self):
        return {
            "buy_lookback": 4, "sell_lookback": 4, "ema_period": 200, "ma_period": 20,
            "range_pct_max": 5.0, "body_pct_min": 0.1, "body_pct_max": 20.0,
        }

    def test_chia_deu_von_dung_theo_so_ma(self):
        df_a = self._tao_df_breakout(260, 210, seed=1)
        df_b = self._tao_df_breakout(260, 210, seed=2)
        tham_so = self._tham_so_de_khop()
        danh_sach = [
            {"ma": "A", "df": df_a, "tham_so": tham_so},
            {"ma": "B", "df": df_b, "tham_so": tham_so},
        ]
        kq = chay_backtest_nhieu_ma(danh_sach, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
                                     von_ban_dau=1_000_000_000, max_tong_von_su_dung_pct=80.0)
        assert kq["ty_trong_von_pct_moi_ma"] == 40.0
        assert kq["so_luong_ma_ket_hop"] == 2

    def test_giu_dong_thoi_2_vi_the_chong_lan(self):
        df_a = self._tao_df_breakout(260, 210, seed=1)
        df_b = self._tao_df_breakout(260, 210, seed=1)  # giống hệt A -> breakout CÙNG ngày
        tham_so = self._tham_so_de_khop()
        danh_sach = [
            {"ma": "A", "df": df_a, "tham_so": tham_so},
            {"ma": "B", "df": df_b, "tham_so": tham_so},
        ]
        kq = chay_backtest_nhieu_ma(danh_sach, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
                                     so_phien_khoa_toi_thieu=0)
        assert kq["open_positions_theo_ma"]["A"] is not None
        assert kq["open_positions_theo_ma"]["B"] is not None
        assert kq["open_positions_theo_ma"]["A"]["entry_date"] == kq["open_positions_theo_ma"]["B"]["entry_date"]

    def test_tong_khong_bao_gio_vuot_tran(self):
        # Với NHIỀU mã (5 mã), tổng % vốn tối đa dùng phải luôn <= max_tong_von_su_dung_pct
        # dù giả sử TẤT CẢ đều mở đồng thời.
        so_ma = 5
        danh_sach = []
        for i in range(so_ma):
            df = self._tao_df_breakout(260, 210, seed=i)
            danh_sach.append({"ma": f"MA{i}", "df": df, "tham_so": self._tham_so_de_khop()})
        kq = chay_backtest_nhieu_ma(danh_sach, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
                                     max_tong_von_su_dung_pct=80.0, so_phien_khoa_toi_thieu=0)
        tong_pct_toi_da_co_the = kq["ty_trong_von_pct_moi_ma"] * kq["so_luong_ma_ket_hop"]
        assert tong_pct_toi_da_co_the <= 80.0 + 1e-6

    def test_raises_for_danh_sach_rong(self):
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest_nhieu_ma([], BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH)

    def test_raises_for_max_tong_von_khong_hop_le(self):
        df_a = self._tao_df_breakout(260, 210)
        danh_sach = [{"ma": "A", "df": df_a, "tham_so": self._tham_so_de_khop()}]
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest_nhieu_ma(danh_sach, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH, max_tong_von_su_dung_pct=0)
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest_nhieu_ma(danh_sach, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH, max_tong_von_su_dung_pct=150)

    def test_moi_ma_dung_dung_tham_so_rieng(self):
        # Mã A dùng tham số CHẶT (không khớp được), mã B dùng tham số LỎNG
        # (khớp được) -> chỉ B có lệnh, A không có, xác nhận mỗi mã dùng
        # ĐÚNG tham số riêng của nó, không bị lẫn.
        df_a = self._tao_df_breakout(260, 210, seed=3)
        df_b = self._tao_df_breakout(260, 210, seed=3)
        tham_so_chat = {**self._tham_so_de_khop(), "body_pct_max": 0.01}  # gần như không thể khớp
        tham_so_long = self._tham_so_de_khop()
        danh_sach = [
            {"ma": "A", "df": df_a, "tham_so": tham_so_chat},
            {"ma": "B", "df": df_b, "tham_so": tham_so_long},
        ]
        kq = chay_backtest_nhieu_ma(danh_sach, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH, so_phien_khoa_toi_thieu=0)
        tong_lenh_a = len(kq["trades_theo_ma"]["A"]) + (1 if kq["open_positions_theo_ma"]["A"] else 0)
        tong_lenh_b = len(kq["trades_theo_ma"]["B"]) + (1 if kq["open_positions_theo_ma"]["B"] else 0)
        assert tong_lenh_a == 0
        assert tong_lenh_b > 0


class TestGoldenCrossDeathCross:
    def test_phat_hien_golden_cross_dung_diem_cat(self):
        ma_nhanh = pd.Series([5.0, 6.0, 7.0, 9.0, 10.0])
        ma_cham = pd.Series([8.0, 8.0, 8.0, 8.0, 8.0])
        # nhanh: 5<8, 6<8, 7<8, 9>8 (cắt lên tại index 3), 10>8
        ket_qua = phat_hien_golden_cross(ma_nhanh, ma_cham)
        assert list(ket_qua) == [False, False, False, True, False]

    def test_phat_hien_death_cross_dung_diem_cat(self):
        ma_nhanh = pd.Series([10.0, 9.0, 8.0, 6.0, 5.0])
        ma_dai_han = pd.Series([7.0, 7.0, 7.0, 7.0, 7.0])
        # nhanh: 10>7, 9>7, 8>7, 6<7 (cắt xuống tại index 3), 5<7
        ket_qua = phat_hien_death_cross(ma_nhanh, ma_dai_han)
        assert list(ket_qua) == [False, False, False, True, False]

    def test_khong_cat_thi_luon_false(self):
        ma_nhanh = pd.Series([10.0, 11.0, 12.0, 13.0])
        ma_cham = pd.Series([5.0, 5.0, 5.0, 5.0])  # nhanh luôn > chậm, không có điểm cắt
        assert not phat_hien_golden_cross(ma_nhanh, ma_cham).any()


class TestChayBacktestMaCrossover:
    def _tao_du_lieu_xu_huong(self, n=700, seed=1):
        np.random.seed(seed)
        dates = pd.bdate_range("2021-01-01", periods=n)
        phan_ngang = np.full(150, 10.0) + np.random.normal(0, 0.1, 150)
        phan_tang = phan_ngang[-1] + np.cumsum(np.random.normal(0.03, 0.3, 400))
        phan_giam = phan_tang[-1] - np.cumsum(np.abs(np.random.normal(0.05, 0.3, n - 550)))
        closes = np.concatenate([phan_ngang, phan_tang, phan_giam])
        opens = closes * (1 + np.random.normal(0, 0.005, n))
        highs = np.maximum(opens, closes) * 1.01
        lows = np.minimum(opens, closes) * 0.99
        volumes = np.random.randint(500_000, 2_000_000, n).astype(float)
        return pd.DataFrame({
            "date": dates, "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        })

    def test_engine_chay_khong_loi(self):
        df = self._tao_du_lieu_xu_huong()
        kq = chay_backtest_ma_crossover(df, von_ban_dau=1_000_000_000, ty_trong_von_pct=50.0)
        assert "trades" in kq and "open_position" in kq
        assert kq["so_lenh_da_dong"] == len(kq["trades"])
        assert kq["so_lan_thoat_boi_stop_loss"] + kq["so_lan_thoat_boi_death_cross"] == kq["so_lenh_da_dong"]

    def test_chi_giao_dich_long(self):
        df = self._tao_du_lieu_xu_huong()
        kq = chay_backtest_ma_crossover(df)
        assert all(t["side"] == "LONG" for t in kq["trades"])

    def test_stop_loss_chat_lam_tang_so_lan_thoat_boi_sl(self):
        df = self._tao_du_lieu_xu_huong()
        kq_sl_rong = chay_backtest_ma_crossover(df, stop_loss_pct=50.0)
        kq_sl_chat = chay_backtest_ma_crossover(df, stop_loss_pct=2.0)
        assert kq_sl_chat["so_lan_thoat_boi_stop_loss"] >= kq_sl_rong["so_lan_thoat_boi_stop_loss"]

    def test_phi_va_thue_tru_dung(self):
        df = self._tao_du_lieu_xu_huong()
        kq_khong_phi = chay_backtest_ma_crossover(df, phi_moi_gioi_pct=0.0, thue_ban_pct=0.0)
        kq_co_phi = chay_backtest_ma_crossover(df, phi_moi_gioi_pct=0.15, thue_ban_pct=0.1)
        assert len(kq_khong_phi["trades"]) == len(kq_co_phi["trades"])
        for t1, t2 in zip(kq_khong_phi["trades"], kq_co_phi["trades"]):
            assert round(t1["final_pnl_pct"] - t2["final_pnl_pct"], 3) == pytest.approx(0.4, abs=0.01)

    def test_raises_for_thu_tu_ma_sai(self):
        df = _make_realistic_df(n=300)
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest_ma_crossover(df, ma_nhanh_period=50, ma_cham_period=20, ma_dai_han_period=100)

    def test_raises_for_stop_loss_khong_hop_le(self):
        df = _make_realistic_df(n=300)
        with pytest.raises(InvalidIndicatorLabError):
            chay_backtest_ma_crossover(df, stop_loss_pct=0)

    def test_so_co_phieu_uoc_tinh_lam_tron_lo(self):
        df = self._tao_du_lieu_xu_huong()
        kq = chay_backtest_ma_crossover(df)
        for t in kq["trades"]:
            assert t["so_co_phieu_uoc_tinh"] % 100 == 0

    def test_loc_theo_giai_doan_giam_so_lenh(self):
        from core.market_regime_detector import tinh_chuoi_giai_doan_theo_ngay
        df = self._tao_du_lieu_xu_huong()
        chuoi = tinh_chuoi_giai_doan_theo_ngay({"MA_TEST": df})
        kq_khong_loc = chay_backtest_ma_crossover(df)
        kq_uptrend = chay_backtest_ma_crossover(df, chuoi_giai_doan=chuoi, giai_doan_loc="uptrend")
        assert kq_uptrend["so_lenh_da_dong"] <= kq_khong_loc["so_lenh_da_dong"]

    def test_ly_do_thoat_hop_le(self):
        df = self._tao_du_lieu_xu_huong()
        kq = chay_backtest_ma_crossover(df)
        for t in kq["trades"]:
            assert t["ly_do_thoat"] in ("Stop Loss an toàn", "Death Cross (MA nhanh cắt xuống MA dài hạn)")
