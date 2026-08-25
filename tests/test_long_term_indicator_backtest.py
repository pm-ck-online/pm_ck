"""
Unit test cho core/long_term_indicator_backtest.py

Theo đúng phong cách tests/test_backtest_engine.py: dùng kịch bản giá THỦ
CÔNG (tính tay kết quả kỳ vọng) cho phần logic mới (bucket theo giai đoạn,
chọn bộ chỉ số tốt nhất); phần chỉ báo kỹ thuật (MA/EMA/RSI/Bollinger) đã
được kiểm chứng riêng ở tests/test_indicators.py nên ở đây chỉ kiểm tra
WIRING (đủ cột, đúng cấu trúc, đúng thời điểm tín hiệu) — không tính lại
công thức chỉ báo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.backtest_engine import Trade
from core.long_term_indicator_backtest import (
    VALID_REGIMES,
    InvalidLongTermBacktestError,
    backtest_theo_giai_doan,
    backtest_toan_bo_8_bo_chi_so,
    tim_bo_chi_so_tot_nhat,
    tinh_chi_bao_dai_han,
    xay_8_bo_chi_so,
)


def _make_df(prices: list[float], volumes: list[float] | None = None, start_date: str = "2024-01-01") -> pd.DataFrame:
    n = len(prices)
    dates = pd.bdate_range(start=start_date, periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": volumes if volumes is not None else [1000] * n,
    })


# ==============================================================================
# tinh_chi_bao_dai_han — kiểm tra WIRING (đủ cột, không tính lại công thức)
# ==============================================================================

class TestTinhChiBaoDaiHan:
    def test_them_du_cot_chi_bao(self):
        df = _make_df(list(range(100, 140)))  # 40 phiên, tăng dần
        ket_qua = tinh_chi_bao_dai_han(df)
        for cot in ["ma20", "ema50", "ema200", "rsi14", "bb_upper", "bb_middle", "bb_lower", "vol_ma20"]:
            assert cot in ket_qua.columns

    def test_thieu_cot_bat_buoc_bao_loi(self):
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5), "close": [1, 2, 3, 4, 5]})
        with pytest.raises(InvalidLongTermBacktestError):
            tinh_chi_bao_dai_han(df)

    def test_ma20_dung_gia_tri_tinh_tay(self):
        # 20 phiên đầu giá 100 (MA20 phiên thứ 20 = 100), phiên 21 giá 200
        # -> MA20 phiên 21 = (19*100 + 200)/20 = 105.
        prices = [100] * 20 + [200]
        df = _make_df(prices)
        ket_qua = tinh_chi_bao_dai_han(df)
        assert ket_qua["ma20"].iloc[19] == pytest.approx(100.0)
        assert ket_qua["ma20"].iloc[20] == pytest.approx(105.0)


# ==============================================================================
# xay_8_bo_chi_so — kiểm tra thời điểm tín hiệu đúng như kỳ vọng
# ==============================================================================

class TestXay8BoChiSo:
    def test_du_8_bo_chi_so(self):
        df = tinh_chi_bao_dai_han(_make_df(list(range(100, 140))))
        bo_chi_so = xay_8_bo_chi_so(df)
        assert len(bo_chi_so) == 8
        for entry, exit_ in bo_chi_so.values():
            assert len(entry) == len(df)
            assert len(exit_) == len(df)
            assert entry.dtype == bool
            assert exit_.dtype == bool

    def test_rsi14_entry_khi_qua_ban_exit_khi_qua_mua(self):
        # Giá giảm mạnh liên tục 20 phiên (RSI thấp) rồi tăng mạnh liên tục
        # 20 phiên (RSI cao) -> phải có ít nhất 1 điểm entry (RSI<30) trong
        # đoạn giảm và ít nhất 1 điểm exit (RSI>70) trong đoạn tăng.
        giam = list(np.linspace(100, 50, 20))
        tang = list(np.linspace(50, 150, 20))
        df = tinh_chi_bao_dai_han(_make_df(giam + tang))
        bo_chi_so = xay_8_bo_chi_so(df)
        entry, exit_ = bo_chi_so["RSI14 (Quá mua/Quá bán 30-70)"]
        assert entry[:20].any(), "Phải có tín hiệu MUA khi RSI rơi xuống vùng quá bán."
        assert exit_[20:].any(), "Phải có tín hiệu BÁN khi RSI tăng lên vùng quá mua."
        # RSI chỉ tính được từ phiên có đủ dữ liệu -> không có tín hiệu ở các phiên đầu chưa đủ dữ liệu
        assert not entry[:14].any()

    def test_bollinger_bounce_entry_tai_diem_gia_thap_nhat(self):
        # Giá dao động quanh 100 rồi rơi mạnh 1 phiên xuống 70 -> phiên đó
        # phải nằm dưới dải Bollinger dưới -> có tín hiệu MUA (bounce).
        prices = [100, 102, 98, 101, 99, 100, 103, 97, 100, 101,
                  100, 99, 102, 98, 100, 101, 100, 99, 100, 101, 70]
        df = tinh_chi_bao_dai_han(_make_df(prices))
        _, _ = df["bb_lower"], df["close"]
        bo_chi_so = xay_8_bo_chi_so(df)
        entry, _ = bo_chi_so["Bollinger Bounce (mua đáy dải dưới)"]
        assert bool(entry.iloc[20]) is True

    def test_volume_breakout_can_ca_gia_va_volume(self):
        prices = [100] * 25 + [110]
        # Volume đột biến CHỈ ở phiên cuối
        volumes = [1000] * 25 + [3000]
        df = tinh_chi_bao_dai_han(_make_df(prices, volumes=volumes))
        bo_chi_so = xay_8_bo_chi_so(df)
        entry, _ = bo_chi_so["Volume Breakout + MA20"]
        assert bool(entry.iloc[25]) is True
        # Phiên trước đó giá chưa vượt MA20 (giá không đổi) -> không có tín hiệu
        assert not entry.iloc[:25].any()

    def test_buy_and_hold_chi_mua_dung_1_lan_o_ngay_dau_du_du_lieu(self):
        df = tinh_chi_bao_dai_han(_make_df(list(np.linspace(100, 300, 220))))
        bo_chi_so = xay_8_bo_chi_so(df)
        entry, exit_ = bo_chi_so["Mua và giữ (Buy & Hold)"]
        assert entry.sum() == 1
        assert not exit_.any()
        vi_tri_mua = int(np.flatnonzero(entry.to_numpy())[0])
        assert df["ema200"].iloc[vi_tri_mua - 1] != df["ema200"].iloc[vi_tri_mua - 1]  # NaN trước đó
        assert not pd.isna(df["ema200"].iloc[vi_tri_mua])


# ==============================================================================
# backtest_theo_giai_doan — kịch bản giá thủ công, tính tay kết quả kỳ vọng
# (mượn đúng kịch bản 5 phiên trong tests/test_backtest_engine.py)
# ==============================================================================

class TestBacktestTheoGiaiDoan:
    """5 phiên, giá: 100, 110, 90, 95, 130. Mua tại open ngày 1 (=110), bán
    tại open ngày 3 (=95), fee=0 -> qty=9090, pnl=-136,350,
    pnl_pct = -136350/999900*100 = -13,6364%. Gán ngày 1 (ngày vào lệnh)
    là "downtrend" -> toàn bộ kết quả phải rơi vào bucket downtrend.
    """

    @pytest.fixture
    def df(self):
        return _make_df([100, 110, 90, 95, 130])

    @pytest.fixture
    def ket_qua(self, df):
        entry = pd.Series([True, False, False, False, False])
        exit_ = pd.Series([False, False, True, False, False])
        regime_series = pd.Series({
            df["date"].iloc[0]: "uptrend",
            df["date"].iloc[1]: "downtrend",
            df["date"].iloc[2]: "downtrend",
            df["date"].iloc[3]: "sideway",
            df["date"].iloc[4]: "sideway",
        })
        return backtest_theo_giai_doan(
            df, regime_series, entry, exit_,
            initial_capital=1_000_000.0, fee_pct=0.0,
        )

    def test_co_du_3_giai_doan(self, ket_qua):
        assert set(ket_qua.keys()) == VALID_REGIMES

    def test_giai_doan_khong_co_lenh(self, ket_qua):
        for regime in ("uptrend", "sideway"):
            assert ket_qua[regime]["n_trades"] == 0
            assert ket_qua[regime]["total_return_pct"] is None
            assert ket_qua[regime]["ending_capital"] == pytest.approx(1_000_000.0)

    def test_giai_doan_downtrend_dung_ket_qua_tinh_tay(self, ket_qua):
        entry_cost = 9090 * 110  # = 999,900
        pnl = 9090 * 95 - entry_cost  # = -136,350
        pnl_pct_ky_vong = pnl / entry_cost * 100

        dt = ket_qua["downtrend"]
        assert dt["n_trades"] == 1
        assert dt["win_rate_pct"] == pytest.approx(0.0)
        assert dt["total_return_pct"] == pytest.approx(pnl_pct_ky_vong, abs=0.01)
        assert dt["avg_return_pct"] == pytest.approx(pnl_pct_ky_vong, abs=0.01)
        assert dt["ending_capital"] == pytest.approx(1_000_000.0 * (1 + pnl_pct_ky_vong / 100), abs=1)


# ==============================================================================
# backtest_toan_bo_8_bo_chi_so — kiểm tra cấu trúc kết quả tổng hợp
# ==============================================================================

class TestBacktestToanBo8BoChiSo:
    def test_cau_truc_ket_qua(self):
        df = _make_df(list(np.linspace(100, 300, 220)))
        regime_series = pd.Series("uptrend", index=df["date"])
        ket_qua = backtest_toan_bo_8_bo_chi_so(df, regime_series, initial_capital=1_000_000.0, fee_pct=0.15)
        assert len(ket_qua) == 8
        for ten_bo_chi_so, ket_qua_giai_doan in ket_qua.items():
            assert set(ket_qua_giai_doan.keys()) == VALID_REGIMES
            for regime_stats in ket_qua_giai_doan.values():
                assert "n_trades" in regime_stats
                assert "ending_capital" in regime_stats


# ==============================================================================
# tim_bo_chi_so_tot_nhat
# ==============================================================================

class TestTimBoChiSoTotNhat:
    def _ket_qua_gia(self, **tong_ln_theo_ten):
        return {
            ten: {
                "uptrend": {"n_trades": 1 if tong_ln is not None else 0, "total_return_pct": tong_ln},
                "sideway": {"n_trades": 0, "total_return_pct": None},
                "downtrend": {"n_trades": 0, "total_return_pct": None},
            }
            for ten, tong_ln in tong_ln_theo_ten.items()
        }

    def test_chon_dung_bo_co_loi_nhuan_cao_nhat(self):
        ket_qua = self._ket_qua_gia(A=10.0, B=25.0, C=-5.0)
        assert tim_bo_chi_so_tot_nhat(ket_qua, "uptrend") == "B"

    def test_tra_ve_none_khi_khong_co_bo_nao_co_lenh(self):
        ket_qua = self._ket_qua_gia(A=None, B=None)
        assert tim_bo_chi_so_tot_nhat(ket_qua, "sideway") is None

    def test_giai_doan_khong_hop_le_bao_loi(self):
        ket_qua = self._ket_qua_gia(A=10.0)
        with pytest.raises(InvalidLongTermBacktestError):
            tim_bo_chi_so_tot_nhat(ket_qua, "khong_ton_tai")
