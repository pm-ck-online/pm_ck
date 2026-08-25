"""Unit test cho core/market_regime_ensemble.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.market_regime_ensemble import (
    NGUONG_BEAR_MARKOV,
    NGUONG_BULL_MARKOV,
    SO_PHIEN_TOI_THIEU_MARKOV,
    TRONG_SO_PHUONG_PHAP,
    phan_tich_ensemble_theo_nhom,
    phan_tich_ensemble_toan_bo,
    phuong_phap_A_breadth,
    phuong_phap_B_peak_trough,
    phuong_phap_C_markov_switching,
    tong_hop_3_phuong_phap,
)


def _make_ohlcv_xu_huong(n=400, log_return_mean=0.0015, log_return_std=0.01, seed=1, gia_ban_dau=1000.0):
    np.random.seed(seed)
    log_returns = np.random.normal(log_return_mean, log_return_std, n)
    closes = gia_ban_dau * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame({
        "date": dates, "close": closes, "open": closes * 0.999,
        "high": closes * 1.005, "low": closes * 0.995, "volume": [1_000_000.0] * n,
    })


class TestPhuongPhapABreadth:
    def test_uptrend_khi_breadth_cao(self):
        snaps = [{"close": 100.0, "ema200": 90.0}] * 70 + [{"close": 80.0, "ema200": 90.0}] * 30
        kq = phuong_phap_A_breadth(snaps)
        assert kq["nhan"] == "UPTREND"
        assert kq["gia_tri_so"] == 70.0

    def test_downtrend_khi_breadth_thap(self):
        snaps = [{"close": 100.0, "ema200": 90.0}] * 20 + [{"close": 80.0, "ema200": 90.0}] * 80
        kq = phuong_phap_A_breadth(snaps)
        assert kq["nhan"] == "DOWNTREND"

    def test_sideway_khi_breadth_giua(self):
        snaps = [{"close": 100.0, "ema200": 90.0}] * 50 + [{"close": 80.0, "ema200": 90.0}] * 50
        kq = phuong_phap_A_breadth(snaps)
        assert kq["nhan"] == "SIDEWAY"

    def test_sideway_khi_khong_co_du_lieu_hop_le(self):
        kq = phuong_phap_A_breadth([{"close": 100.0, "ema200": None}])
        assert kq["nhan"] == "SIDEWAY"
        assert kq["gia_tri_so"] is None


class TestPhuongPhapBPeakTrough:
    def test_sideway_khi_thieu_du_lieu(self):
        kq = phuong_phap_B_peak_trough(None)
        assert kq["nhan"] == "SIDEWAY"

    def test_sideway_khi_df_qua_ngan(self):
        df = _make_ohlcv_xu_huong(n=10)
        kq = phuong_phap_B_peak_trough(df)
        assert kq["nhan"] == "SIDEWAY"

    def test_uptrend_higher_high_higher_low(self):
        df = _make_ohlcv_xu_huong(n=400, log_return_mean=0.002, log_return_std=0.012, seed=10)
        kq = phuong_phap_B_peak_trough(df)
        assert kq["nhan"] in ("UPTREND", "SIDEWAY")  # xu hướng tăng -> khả năng cao UPTREND, không loại trừ SIDEWAY do nhiễu


class TestPhuongPhapCMarkov:
    def test_sideway_khi_thieu_du_lieu(self):
        kq = phuong_phap_C_markov_switching(None)
        assert kq["nhan"] == "SIDEWAY"
        assert "Chưa đủ dữ liệu" in kq["chi_tiet"]

    def test_sideway_khi_chua_du_so_phien_toi_thieu(self):
        df = _make_ohlcv_xu_huong(n=SO_PHIEN_TOI_THIEU_MARKOV - 10)
        kq = phuong_phap_C_markov_switching(df)
        assert kq["nhan"] == "SIDEWAY"

    def test_downtrend_ro_rang(self):
        df = _make_ohlcv_xu_huong(n=400, log_return_mean=-0.004, log_return_std=0.015, seed=2)
        kq = phuong_phap_C_markov_switching(df)
        assert kq["nhan"] == "DOWNTREND"
        assert kq["gia_tri_so"] <= NGUONG_BEAR_MARKOV

    def test_uptrend_ro_rang(self):
        df = _make_ohlcv_xu_huong(n=400, log_return_mean=0.004, log_return_std=0.015, seed=3)
        kq = phuong_phap_C_markov_switching(df)
        assert kq["nhan"] == "UPTREND"
        assert kq["gia_tri_so"] >= NGUONG_BULL_MARKOV

    def test_gia_tri_so_trong_khoang_0_1(self):
        df = _make_ohlcv_xu_huong(n=400, seed=4)
        kq = phuong_phap_C_markov_switching(df)
        if kq["gia_tri_so"] is not None:
            assert 0.0 <= kq["gia_tri_so"] <= 1.0


class TestTongHop3PhuongPhap:
    def test_dong_thuan_3_3_do_tin_cay_cao(self):
        kq = tong_hop_3_phuong_phap(
            {"nhan": "UPTREND"}, {"nhan": "UPTREND"}, {"nhan": "UPTREND"},
        )
        assert kq["nhan_tong_hop"] == "UPTREND"
        assert kq["do_tin_cay"] == "CAO"
        assert kq["so_phuong_phap_dong_thuan"] == 3

    def test_dong_thuan_2_3_do_tin_cay_trung_binh(self):
        kq = tong_hop_3_phuong_phap(
            {"nhan": "UPTREND"}, {"nhan": "SIDEWAY"}, {"nhan": "UPTREND"},
        )
        assert kq["nhan_tong_hop"] == "UPTREND"
        assert kq["do_tin_cay"] == "TRUNG_BINH"
        assert kq["so_phuong_phap_dong_thuan"] == 2

    def test_ca_3_khac_nhau_dung_trong_so_pha_the_be_tac(self):
        # A=UPTREND (0.4), B=DOWNTREND (0.25), C=SIDEWAY (0.35) -> A thắng vì trọng số cao nhất
        kq = tong_hop_3_phuong_phap(
            {"nhan": "UPTREND"}, {"nhan": "DOWNTREND"}, {"nhan": "SIDEWAY"},
        )
        assert kq["nhan_tong_hop"] == "UPTREND"
        assert kq["do_tin_cay"] == "THAP"
        assert kq["so_phuong_phap_dong_thuan"] == 1

    def test_tong_trong_so_bang_1(self):
        assert sum(TRONG_SO_PHUONG_PHAP.values()) == pytest.approx(1.0)

    def test_phieu_bau_chi_tiet_dung(self):
        kq = tong_hop_3_phuong_phap(
            {"nhan": "UPTREND"}, {"nhan": "SIDEWAY"}, {"nhan": "DOWNTREND"},
        )
        assert kq["phieu_bau_chi_tiet"] == {"A": "UPTREND", "B": "SIDEWAY", "C": "DOWNTREND"}


class TestPhanTichEnsembleTheoNhom:
    def test_tra_ve_du_cac_truong(self):
        snaps = [{"close": 100.0, "ema200": 90.0}] * 70 + [{"close": 80.0, "ema200": 90.0}] * 30
        df = _make_ohlcv_xu_huong(n=100)
        kq = phan_tich_ensemble_theo_nhom("Toàn thị trường", snaps, df)
        for truong in ("nhom", "phuong_phap_A_breadth", "phuong_phap_B_peak_trough",
                       "phuong_phap_C_markov", "KET_LUAN_TONG_HOP", "do_tin_cay", "chi_tiet"):
            assert truong in kq
        assert kq["nhom"] == "Toàn thị trường"


class TestPhanTichEnsembleToanBo:
    def test_tra_ve_dataframe_dung_cau_truc(self):
        snaps = [{"close": 100.0, "ema200": 90.0}] * 70 + [{"close": 80.0, "ema200": 90.0}] * 30
        df = _make_ohlcv_xu_huong(n=100)
        df_ket_qua = phan_tich_ensemble_toan_bo(
            snaps, df, {"Ngân hàng": (snaps, df), "Chứng khoán": (snaps, df)},
        )
        assert isinstance(df_ket_qua, pd.DataFrame)
        assert len(df_ket_qua) == 3  # Toàn thị trường + 2 ngành
        assert list(df_ket_qua.columns) == [
            "nhom", "phuong_phap_A_breadth", "phuong_phap_B_peak_trough",
            "phuong_phap_C_markov", "KET_LUAN_TONG_HOP", "do_tin_cay",
        ]
        assert df_ket_qua["nhom"].iloc[0] == "Toàn thị trường"
        assert set(df_ket_qua["nhom"].iloc[1:]) == {"Ngân hàng", "Chứng khoán"}

    def test_hoat_dong_dung_voi_danh_sach_nganh_rong(self):
        snaps = [{"close": 100.0, "ema200": 90.0}]
        df = _make_ohlcv_xu_huong(n=100)
        df_ket_qua = phan_tich_ensemble_toan_bo(snaps, df, {})
        assert len(df_ket_qua) == 1
        assert df_ket_qua["nhom"].iloc[0] == "Toàn thị trường"


class TestDungChiSoDaiDienTuGiaDongCua:
    def test_chuan_hoa_va_lay_trung_binh_dung(self):
        from core.market_regime_ensemble import dung_chi_so_dai_dien_tu_gia_dong_cua
        dates = pd.bdate_range("2024-01-01", periods=30)
        # Mã A: giá 100->200 (tăng gấp đôi), Mã B: giá 10->20 (cũng tăng gấp đôi)
        # Sau chuẩn hóa cả 2 đều tăng từ 100->200 -> trung bình phải = từng mã.
        series_a = pd.Series(np.linspace(100, 200, 30), index=dates)
        series_b = pd.Series(np.linspace(10, 20, 30), index=dates)
        df_ket_qua = dung_chi_so_dai_dien_tu_gia_dong_cua([series_a, series_b])
        assert df_ket_qua is not None
        assert df_ket_qua["close"].iloc[0] == pytest.approx(100.0, abs=0.1)
        assert df_ket_qua["close"].iloc[-1] == pytest.approx(200.0, abs=0.1)

    def test_tra_ve_none_khi_danh_sach_rong(self):
        from core.market_regime_ensemble import dung_chi_so_dai_dien_tu_gia_dong_cua
        assert dung_chi_so_dai_dien_tu_gia_dong_cua([]) is None

    def test_bo_qua_chuoi_qua_ngan(self):
        from core.market_regime_ensemble import dung_chi_so_dai_dien_tu_gia_dong_cua
        series_ngan = pd.Series([100.0, 101.0], index=pd.bdate_range("2024-01-01", periods=2))
        assert dung_chi_so_dai_dien_tu_gia_dong_cua([series_ngan]) is None

    def test_co_du_cot_ohlcv(self):
        from core.market_regime_ensemble import dung_chi_so_dai_dien_tu_gia_dong_cua
        dates = pd.bdate_range("2024-01-01", periods=30)
        series_a = pd.Series(np.linspace(100, 200, 30), index=dates)
        df_ket_qua = dung_chi_so_dai_dien_tu_gia_dong_cua([series_a])
        for col in ("date", "open", "high", "low", "close", "volume"):
            assert col in df_ket_qua.columns


# ==============================================================================
# tinh_chuoi_ensemble_theo_ngay — walk-forward theo từng ngày cho 1 mã
# ==============================================================================
#
# Dùng ngưỡng Markov THẤP HƠN mặc định (60 thay vì 250 phiên) qua tham số
# `config` — để test chạy nhanh (fit Markov trên chuỗi ngắn hơn nhiều lần)
# mà vẫn kiểm chứng đúng cơ chế walk-forward + refit theo chu kỳ, không
# phải cơ chế "250 phiên" (đã có hằng số SO_PHIEN_TOI_THIEU_MARKOV riêng).

CONFIG_TEST_NHANH = {"markov_so_phien_toi_thieu": 60, "markov_refit_every": 10}


class TestTinhChuoiEnsembleTheoNgay:
    def test_qua_ngan_tra_ve_series_rong(self):
        from core.market_regime_ensemble import tinh_chuoi_ensemble_theo_ngay
        df = _make_ohlcv_xu_huong(n=30)
        ket_qua = tinh_chuoi_ensemble_theo_ngay(df, config=CONFIG_TEST_NHANH)
        assert isinstance(ket_qua, pd.Series)
        assert ket_qua.empty

    def test_cau_truc_va_do_dai_ket_qua(self):
        from core.market_regime_ensemble import tinh_chuoi_ensemble_theo_ngay
        n = 90
        df = _make_ohlcv_xu_huong(n=n, log_return_mean=0.003, seed=2)
        ket_qua = tinh_chuoi_ensemble_theo_ngay(df, config=CONFIG_TEST_NHANH)

        so_phien_toi_thieu = CONFIG_TEST_NHANH["markov_so_phien_toi_thieu"]
        assert isinstance(ket_qua, pd.Series)
        assert len(ket_qua) == n - so_phien_toi_thieu + 1
        assert set(ket_qua.unique()) <= {"uptrend", "sideway", "downtrend"}
        assert isinstance(ket_qua.index, pd.DatetimeIndex)
        # Ngày đầu tiên trong chuỗi phải đúng là phiên thứ `so_phien_toi_thieu`
        assert ket_qua.index[0] == df["date"].iloc[so_phien_toi_thieu - 1]

    def test_chi_fit_lai_markov_theo_dung_chu_ky(self, monkeypatch):
        """Kiểm chứng cơ chế MỚI (refit theo chu kỳ, không phải mỗi ngày)
        bằng cách thay `phuong_phap_C_markov_switching` bằng 1 hàm giả đếm
        số lần gọi + độ dài cửa sổ mỗi lần gọi — độc lập với việc mô hình
        Markov thật có hội tụ hay không (tránh test bị "flaky").
        """
        from core import market_regime_ensemble as mre

        do_dai_cua_so_da_goi: list[int] = []

        def markov_gia(df_con, so_phien_toi_thieu):
            do_dai_cua_so_da_goi.append(len(df_con))
            return {"nhan": "UPTREND", "chi_tiet": "gia", "gia_tri_so": 0.9}

        monkeypatch.setattr(mre, "phuong_phap_C_markov_switching", markov_gia)

        df = _make_ohlcv_xu_huong(n=90, seed=5)
        mre.tinh_chuoi_ensemble_theo_ngay(df, config=CONFIG_TEST_NHANH)

        # so_phien_toi_thieu=60, markov_refit_every=10, tổng 90 phiên ->
        # CHỈ fit lại ở độ dài cửa sổ 60, 70, 80, 90 (4 lần, KHÔNG PHẢI 31
        # lần nếu fit mỗi ngày).
        assert do_dai_cua_so_da_goi == [60, 70, 80, 90]

    def test_doi_chieu_ngay_cuoi_voi_tinh_truc_tiep(self):
        """Nhãn ở NGÀY CUỐI của chuỗi walk-forward phải khớp với việc gọi
        trực tiếp 3 phương pháp + tổng hợp trên TOÀN BỘ dữ liệu (đúng bản
        chất walk-forward: ngày cuối cùng thấy đủ dữ liệu như 1 lần tính
        trực tiếp trên toàn bộ chuỗi).
        """
        from core.indicators import calculate_ema
        from core.market_regime_ensemble import tinh_chuoi_ensemble_theo_ngay

        df = _make_ohlcv_xu_huong(n=90, log_return_mean=0.001, seed=4)
        ket_qua = tinh_chuoi_ensemble_theo_ngay(df, config=CONFIG_TEST_NHANH)

        so_phien_toi_thieu = CONFIG_TEST_NHANH["markov_so_phien_toi_thieu"]
        ema200 = calculate_ema(df, 200)
        snap = [{"close": float(df["close"].iloc[-1]), "ema200": (float(ema200.iloc[-1]) if pd.notna(ema200.iloc[-1]) else None)}]
        ket_qua_A = phuong_phap_A_breadth(snap)
        ket_qua_B = phuong_phap_B_peak_trough(df, so_chu_ky_xet=2)
        ket_qua_C = phuong_phap_C_markov_switching(df[["close"]], so_phien_toi_thieu=so_phien_toi_thieu)
        tong_hop = tong_hop_3_phuong_phap(ket_qua_A, ket_qua_B, ket_qua_C)

        assert ket_qua.iloc[-1] == tong_hop["nhan_tong_hop"].lower()
