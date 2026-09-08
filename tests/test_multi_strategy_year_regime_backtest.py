"""Test cho core/multi_strategy_year_regime_backtest.py — mọi giá trị kỳ
vọng TÍNH TAY theo đúng kỷ luật test của dự án (CLAUDE.md mục 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.backtest_engine import run_backtest
from core.long_term_indicator_backtest import tinh_chi_bao_dai_han, xay_8_bo_chi_so
from core.multi_strategy_year_regime_backtest import (
    _tim_giai_doan_tai_ngay,
    _tong_hop_von_tuan_tu,
    backtest_to_hop_theo_nam_giai_doan,
    loc_ket_qua_theo_dieu_kien,
    xay_to_hop_cap_bo_chi_so,
)


# ==============================================================================
# Test: xay_to_hop_cap_bo_chi_so
# ==============================================================================

class TestXayToHopCapBoChiSo:
    def test_tra_ve_dung_21_to_hop_tu_8_bo(self):
        gia_tri_gia = pd.Series([True, False, True])
        bo_8_gia = {
            f"Bo {i}": (gia_tri_gia, gia_tri_gia) for i in range(7)
        }
        bo_8_gia["Mua và giữ (Buy & Hold)"] = (gia_tri_gia, gia_tri_gia)

        to_hop = xay_to_hop_cap_bo_chi_so(bo_8_gia)

        assert len(to_hop) == 21  # C(7,2)
        assert all("Mua và giữ (Buy & Hold)" not in ten for ten in to_hop)

    def test_entry_la_and_exit_la_or(self):
        entry_a = pd.Series([True, True, False, False])
        exit_a = pd.Series([False, False, True, False])
        entry_b = pd.Series([True, False, True, False])
        exit_b = pd.Series([False, True, False, False])
        bo_8_gia = {
            "A": (entry_a, exit_a), "B": (entry_b, exit_b),
            "Mua và giữ (Buy & Hold)": (entry_a, exit_a),
        }
        to_hop = xay_to_hop_cap_bo_chi_so(bo_8_gia)

        assert len(to_hop) == 1
        entry_combo, exit_combo = to_hop["A + B"]
        assert list(entry_combo) == [True, False, False, False]  # AND
        assert list(exit_combo) == [False, True, True, False]    # OR


# ==============================================================================
# Test: _tim_giai_doan_tai_ngay
# ==============================================================================

class TestTimGiaiDoanTaiNgay:
    def _chuoi(self):
        return pd.Series(
            ["uptrend", "downtrend", "sideway"],
            index=pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"]),
        )

    def test_dung_ngay_khop_chinh_xac(self):
        assert _tim_giai_doan_tai_ngay("2024-01-05", self._chuoi()) == "downtrend"

    def test_ngay_giua_2_moc_lay_moc_gan_nhat_truoc_do(self):
        assert _tim_giai_doan_tai_ngay("2024-01-07", self._chuoi()) == "downtrend"

    def test_ngay_som_hon_toan_bo_chuoi_tra_ve_none(self):
        assert _tim_giai_doan_tai_ngay("2023-12-01", self._chuoi()) is None

    def test_chuoi_rong_tra_ve_none(self):
        assert _tim_giai_doan_tai_ngay("2024-01-01", pd.Series(dtype=object)) is None


# ==============================================================================
# Test: _tong_hop_von_tuan_tu
# ==============================================================================

class _TradeGia:
    def __init__(self, pnl_pct):
        self.pnl_pct = pnl_pct
        self.pnl = pnl_pct  # dấu giống pnl_pct là đủ cho mục đích test (chỉ cần đúng dấu)


class TestTongHopVonTuanTu:
    def test_compounding_va_win_rate_dung(self):
        trades = [_TradeGia(10.0), _TradeGia(-5.0), _TradeGia(20.0)]
        ket_qua = _tong_hop_von_tuan_tu(trades, initial_capital=1_000_000_000.0)

        von_ky_vong = 1_000_000_000.0 * 1.10 * 0.95 * 1.20
        assert ket_qua["n_trades"] == 3
        assert ket_qua["win_rate_pct"] == pytest.approx(200.0 / 3)
        assert ket_qua["total_return_pct"] == pytest.approx((von_ky_vong / 1_000_000_000.0 - 1) * 100.0)


# ==============================================================================
# Test: loc_ket_qua_theo_dieu_kien
# ==============================================================================

class TestLocKetQuaTheoDieuKien:
    def _du_lieu(self):
        return [
            {"ma": "HDB", "ten_bo_chi_so": "MA20", "nam": 2025, "giai_doan_chinh": "uptrend", "n_trades": 10, "total_return_pct": 34.85},
            {"ma": "SSI", "ten_bo_chi_so": "MA20", "nam": 2025, "giai_doan_chinh": "uptrend", "n_trades": 11, "total_return_pct": 47.89},
            {"ma": "SSI", "ten_bo_chi_so": "RSI14", "nam": 2023, "giai_doan_chinh": None, "n_trades": 1, "total_return_pct": 36.34},
            {"ma": "GMD", "ten_bo_chi_so": "MA20", "nam": 2026, "giai_doan_chinh": "downtrend", "n_trades": 8, "total_return_pct": -16.88},
        ]

    def test_loc_theo_nguong_lai(self):
        ket_qua = loc_ket_qua_theo_dieu_kien(self._du_lieu(), nguong_lai_pct=40.0)
        assert [h["ma"] for h in ket_qua] == ["SSI"]
        assert ket_qua[0]["ten_bo_chi_so"] == "MA20"

    def test_nguong_la_lon_hon_khong_phai_lon_hon_bang(self):
        ket_qua = loc_ket_qua_theo_dieu_kien(self._du_lieu(), nguong_lai_pct=34.85)
        assert all(h["total_return_pct"] > 34.85 for h in ket_qua)
        # SSI MA20 (47.89) va SSI RSI14 (36.34) qua nguong; HDB (34.85,
        # dung bang nguong) va GMD (-16.88) bi loai.
        assert len(ket_qua) == 2

    def test_loc_theo_bo_chi_so(self):
        ket_qua = loc_ket_qua_theo_dieu_kien(self._du_lieu(), ten_bo_chi_so="MA20")
        assert len(ket_qua) == 3
        assert all(h["ten_bo_chi_so"] == "MA20" for h in ket_qua)

    def test_loc_theo_giai_doan(self):
        ket_qua = loc_ket_qua_theo_dieu_kien(self._du_lieu(), giai_doan="uptrend")
        assert len(ket_qua) == 2

    def test_loc_theo_nam(self):
        ket_qua = loc_ket_qua_theo_dieu_kien(self._du_lieu(), nam=2026)
        assert len(ket_qua) == 1
        assert ket_qua[0]["ma"] == "GMD"

    def test_loc_theo_so_lenh_toi_thieu(self):
        ket_qua = loc_ket_qua_theo_dieu_kien(self._du_lieu(), so_lenh_toi_thieu=5)
        assert len(ket_qua) == 3  # loai bo dong RSI14 chi 1 lenh

    def test_ket_hop_nhieu_tieu_chi(self):
        ket_qua = loc_ket_qua_theo_dieu_kien(
            self._du_lieu(), ten_bo_chi_so="MA20", giai_doan="uptrend", nguong_lai_pct=30.0,
        )
        assert len(ket_qua) == 2  # HDB va SSI, ca 2 deu MA20+uptrend+>30%

    def test_khong_tieu_chi_nao_tra_ve_nguyen_ban(self):
        du_lieu = self._du_lieu()
        assert loc_ket_qua_theo_dieu_kien(du_lieu) == du_lieu


# ==============================================================================
# Test: backtest_to_hop_theo_nam_giai_doan — kiểm tra CẤU TRÚC + đối
# chiếu chéo với việc tự backtest 1 bộ chỉ số cụ thể bằng run_backtest()
# trực tiếp (đảm bảo không có sai lệch trong khâu bucket theo năm).
# ==============================================================================

def _tao_gia_xu_huong_tang(seed: int, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    closes = 20.0 + np.cumsum(rng.normal(0.05, 0.5, n))
    closes = np.maximum(closes, 1.0)
    return pd.DataFrame({
        "date": dates, "open": closes, "high": closes * 1.01,
        "low": closes * 0.99, "close": closes, "volume": [1_000_000.0] * n,
    })


class TestBacktestToHopTheoNamGiaiDoan:
    def test_cau_truc_ket_qua_va_doi_chieu_cheo_voi_run_backtest_truc_tiep(self):
        df = _tao_gia_xu_huong_tang(seed=42)
        # Giai đoạn giả định: nửa đầu "uptrend", nửa sau "downtrend" — chỉ
        # cần đủ để kiểm tra field "giai_doan_chinh" được gán, không cần
        # phản ánh giai đoạn thị trường thật.
        diem_giua = len(df) // 2
        regime_series = pd.Series(
            ["uptrend"] * diem_giua + ["downtrend"] * (len(df) - diem_giua),
            index=pd.to_datetime(df["date"]),
        )

        ket_qua = backtest_to_hop_theo_nam_giai_doan(df, regime_series, initial_capital=1_000_000_000.0)

        assert isinstance(ket_qua, list)
        assert len(ket_qua) > 0
        cac_khoa_bat_buoc = {
            "ten_bo_chi_so", "nam", "giai_doan_chinh", "so_lenh_giai_doan_chinh",
            "tong_so_lenh_co_giai_doan", "n_trades", "win_rate_pct", "total_return_pct",
        }
        for hang in ket_qua:
            assert cac_khoa_bat_buoc.issubset(hang.keys())
            assert hang["n_trades"] >= 1
            assert hang["giai_doan_chinh"] in ("uptrend", "downtrend", None)

        # Đối chiếu chéo: tự backtest "MA20 (Giá cắt MA20)" TRỰC TIẾP, bucket
        # theo năm bằng ĐÚNG công thức compounding, so với kết quả hàm trả về.
        df_bt = tinh_chi_bao_dai_han(df)
        bo_8 = xay_8_bo_chi_so(df_bt)
        entry_ma20, exit_ma20 = bo_8["MA20 (Giá cắt MA20)"]
        result = run_backtest(
            df_bt, entry_signal_fn=lambda _df: entry_ma20, exit_signal_fn=lambda _df: exit_ma20,
            initial_cash=1_000_000_000.0, fee_pct=0.15,
        )
        if result.trades:
            nam_dau = pd.Timestamp(result.trades[0].entry_date).year
            trades_nam_dau = [t for t in result.trades if pd.Timestamp(t.entry_date).year == nam_dau]
            doi_chieu = _tong_hop_von_tuan_tu(trades_nam_dau, 1_000_000_000.0)

            hang_tuong_ung = [
                h for h in ket_qua if h["ten_bo_chi_so"] == "MA20 (Giá cắt MA20)" and h["nam"] == nam_dau
            ]
            assert len(hang_tuong_ung) == 1
            assert hang_tuong_ung[0]["total_return_pct"] == pytest.approx(doi_chieu["total_return_pct"])
            assert hang_tuong_ung[0]["n_trades"] == doi_chieu["n_trades"]

    def test_khong_co_giai_doan_du_lieu_thi_giai_doan_chinh_la_none(self):
        df = _tao_gia_xu_huong_tang(seed=7)
        regime_rong = pd.Series(dtype=object)  # không có dữ liệu giai đoạn nào

        ket_qua = backtest_to_hop_theo_nam_giai_doan(df, regime_rong, initial_capital=1_000_000_000.0)

        assert len(ket_qua) > 0
        assert all(h["giai_doan_chinh"] is None for h in ket_qua)
        assert all(h["tong_so_lenh_co_giai_doan"] == 0 for h in ket_qua)
