"""Unit test cho main.run_market_regime_ensemble_step() (bổ sung 06/08/2026)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.storage import Storage
from main import run_market_regime_ensemble_step


def _make_ohlcv_records(n=400, seed=1, xu_huong=0.001):
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    closes = 100 * np.exp(np.cumsum(np.random.normal(xu_huong, 0.01, n)))
    return pd.DataFrame({
        "date": dates.astype(str), "open": closes * 0.999, "high": closes * 1.005,
        "low": closes * 0.995, "close": closes, "volume": [1_000_000.0] * n,
    }).to_dict("records")


@pytest.fixture
def storage_voi_du_lieu():
    s = Storage(db_path=":memory:")
    s.save("ohlcv_history", "VNINDEX", {"records": _make_ohlcv_records(seed=99, xu_huong=0.001)})
    s.save("ohlcv_history", "A", {"records": _make_ohlcv_records(seed=1, xu_huong=0.002)})
    s.save("ohlcv_history", "B", {"records": _make_ohlcv_records(seed=2, xu_huong=0.0015)})
    s.save("ohlcv_history", "C", {"records": _make_ohlcv_records(seed=3, xu_huong=-0.001)})
    s.save("symbol_sector", "A", {"sector": "banking"})
    s.save("symbol_sector", "B", {"sector": "banking"})
    s.save("symbol_sector", "C", {"sector": "steel"})
    s.save("indicator_snapshot", "A", {"close": 150.0, "ema200": 100.0})
    s.save("indicator_snapshot", "B", {"close": 140.0, "ema200": 100.0})
    s.save("indicator_snapshot", "C", {"close": 80.0, "ema200": 100.0})
    yield s
    s.close()


class TestRunMarketRegimeEnsembleStep:
    def test_luu_duoc_ket_qua(self, storage_voi_du_lieu):
        run_market_regime_ensemble_step(storage_voi_du_lieu)
        rec = storage_voi_du_lieu.get_latest("market_regime_ensemble", "bang_ket_qua")
        assert rec is not None
        assert "records" in rec["data"]
        assert "tinh_luc" in rec["data"]

    def test_co_du_toan_thi_truong_va_tung_nganh(self, storage_voi_du_lieu):
        run_market_regime_ensemble_step(storage_voi_du_lieu)
        rec = storage_voi_du_lieu.get_latest("market_regime_ensemble", "bang_ket_qua")
        nhom_list = [r["nhom"] for r in rec["data"]["records"]]
        assert "Toàn thị trường" in nhom_list
        assert "banking" in nhom_list
        assert "steel" in nhom_list

    def test_moi_dong_co_du_cot(self, storage_voi_du_lieu):
        run_market_regime_ensemble_step(storage_voi_du_lieu)
        rec = storage_voi_du_lieu.get_latest("market_regime_ensemble", "bang_ket_qua")
        for r in rec["data"]["records"]:
            for col in ("nhom", "phuong_phap_A_breadth", "phuong_phap_B_peak_trough",
                        "phuong_phap_C_markov", "KET_LUAN_TONG_HOP", "do_tin_cay"):
                assert col in r

    def test_khong_loi_khi_khong_co_du_lieu(self):
        s = Storage(db_path=":memory:")
        run_market_regime_ensemble_step(s)  # không raise lỗi, chỉ đơn giản không lưu gì
        assert s.get_latest("market_regime_ensemble", "bang_ket_qua") is None
        s.close()
