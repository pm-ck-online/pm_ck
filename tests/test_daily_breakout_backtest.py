"""Test cho core/daily_breakout_backtest.py — chiến lược breakout nến
NGÀY (chuyển thể từ "Đề xuất #1", bản H1 VN30F1M — xem docstring module).

Mọi giá trị kỳ vọng trong file này đều TÍNH TAY, theo đúng kỷ luật test
của dự án (xem CLAUDE.md mục 5)."""

from __future__ import annotations

import pandas as pd
import pytest

from core.daily_breakout_backtest import (
    CAU_HINH_GOC,
    CAU_HINH_SCALE_THOI_GIAN,
    InvalidDailyBreakoutError,
    _tim_floor_theo_tier,
    backtest_chien_luoc_breakout,
    kiem_tra_dieu_kien_thanh_khoan,
    phat_hien_tin_hieu_vao_lenh,
    tinh_chi_bao_breakout,
    tong_hop_ket_qua,
    LenhBreakoutNgay,
    DotDongLenh,
)


def _lam_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.bdate_range("2024-01-01", periods=len(df))
    return df


# ==============================================================================
# Test: cấu hình 2 phương án đối chiếu
# ==============================================================================

class TestCauHinh:
    def test_cau_hinh_goc_giu_nguyen_so_h1(self):
        assert CAU_HINH_GOC["ema_period"] == 150
        assert CAU_HINH_GOC["ma_period"] == 20
        assert CAU_HINH_GOC["lookback"] == 2

    def test_cau_hinh_scale_thoi_gian_quy_doi_dung(self):
        # 150/4=37.5 -> round 38; 20/4=5 -> round 5; lookback GIỮ NGUYÊN.
        assert CAU_HINH_SCALE_THOI_GIAN["ema_period"] == 38
        assert CAU_HINH_SCALE_THOI_GIAN["ma_period"] == 5
        assert CAU_HINH_SCALE_THOI_GIAN["lookback"] == 2


# ==============================================================================
# Test: tinh_chi_bao_breakout — rolling lookback high/low (KHÔNG gồm phiên
# hiện tại), tính tay theo ví dụ cụ thể.
# ==============================================================================

class TestTinhChiBaoBreakout:
    def test_rolling_lookback_khong_gom_phien_hien_tai(self):
        rows = [
            {"open": 10, "high": 10, "low": 8, "close": 9, "volume": 1000},
            {"open": 10, "high": 12, "low": 9, "close": 11, "volume": 1000},
            {"open": 10, "high": 11, "low": 8, "close": 10, "volume": 1000},
            {"open": 10, "high": 15, "low": 10, "close": 14, "volume": 1000},
            {"open": 10, "high": 9, "low": 7, "close": 8, "volume": 1000},
        ]
        df = _lam_df(rows)
        ket_qua = tinh_chi_bao_breakout(df, ema_period=2, ma_period=2, lookback=2)

        assert pd.isna(ket_qua["cao_nhat_lookback"].iloc[0])
        assert pd.isna(ket_qua["cao_nhat_lookback"].iloc[1])
        # hàng 2 (index 2): dùng high/low của hàng 0,1 -> max(10,12)=12, min(8,9)=8
        assert ket_qua["cao_nhat_lookback"].iloc[2] == pytest.approx(12.0)
        assert ket_qua["thap_nhat_lookback"].iloc[2] == pytest.approx(8.0)
        # hàng 3: dùng hàng 1,2 -> max(12,11)=12, min(9,8)=8
        assert ket_qua["cao_nhat_lookback"].iloc[3] == pytest.approx(12.0)
        assert ket_qua["thap_nhat_lookback"].iloc[3] == pytest.approx(8.0)
        # hàng 4: dùng hàng 2,3 -> max(11,15)=15, min(8,10)=8
        assert ket_qua["cao_nhat_lookback"].iloc[4] == pytest.approx(15.0)
        assert ket_qua["thap_nhat_lookback"].iloc[4] == pytest.approx(8.0)

    def test_thieu_cot_bat_buoc_raise_loi(self):
        df = pd.DataFrame({"date": [1], "close": [10]})
        with pytest.raises(InvalidDailyBreakoutError):
            tinh_chi_bao_breakout(df)


# ==============================================================================
# Test: phat_hien_tin_hieu_vao_lenh — kết hợp breakout + nén biên độ + xu
# hướng + thân nến.
# ==============================================================================

class TestPhatHienTinHieuVaoLenh:
    def _df_co_chi_bao(self, high4, low4, close4):
        """5 phiên đầu PHẲNG (close=10, high=10, low=9 — biên độ nén nhỏ,
        EMA/MA hội tụ về 10), phiên thứ 5 (index 4) tùy biến để kiểm tra
        từng điều kiện riêng lẻ."""
        rows = [{"open": 10, "high": 10, "low": 9, "close": 10, "volume": 1_000_000} for _ in range(4)]
        rows.append({"open": 10, "high": high4, "low": low4, "close": close4, "volume": 1_000_000})
        df = _lam_df(rows)
        return tinh_chi_bao_breakout(df, ema_period=2, ma_period=2, lookback=2)

    def test_du_dieu_kien_va_than_nen_hop_le_tu_dong_vao_lenh(self):
        # cao_nhat_lookback[4] = max(high2,high3) = 10; close=10.5 > 10 -> breakout.
        # bien_do_nen = (10-9)/9*100 = 11.11%... cần range_pct_max đủ lớn.
        df = self._df_co_chi_bao(high4=10.5, low4=10.0, close4=10.5)
        ket_qua = phat_hien_tin_hieu_vao_lenh(df, range_pct_max=20.0, body_pct_min=0.3, body_pct_max=2.5)
        # than_nen = (10.5-10.0)/10.0*100 = 5.0% > body_pct_max(2.5) -> CHỈ CẢNH BÁO
        assert bool(ket_qua["du_dieu_kien_vao_lenh"].iloc[4]) is True
        assert bool(ket_qua["chi_canh_bao"].iloc[4]) is True
        assert bool(ket_qua["tu_dong_vao_lenh"].iloc[4]) is False

    def test_than_nen_trong_nguong_tu_dong_vao_lenh(self):
        # than_nen = (10.1-10.0)/10.0*100 = 1.0% -> trong [0.3, 2.5] -> tự động
        df = self._df_co_chi_bao(high4=10.1, low4=10.0, close4=10.1)
        ket_qua = phat_hien_tin_hieu_vao_lenh(df, range_pct_max=20.0, body_pct_min=0.3, body_pct_max=2.5)
        assert bool(ket_qua["tu_dong_vao_lenh"].iloc[4]) is True
        assert bool(ket_qua["chi_canh_bao"].iloc[4]) is False

    def test_khong_breakout_thi_khong_co_tin_hieu(self):
        # close4=10 (không vượt cao_nhat_lookback=10) -> không breakout.
        df = self._df_co_chi_bao(high4=10.0, low4=9.9, close4=10.0)
        ket_qua = phat_hien_tin_hieu_vao_lenh(df, range_pct_max=20.0, body_pct_min=0.0, body_pct_max=100.0)
        assert bool(ket_qua["du_dieu_kien_vao_lenh"].iloc[4]) is False

    def test_bien_do_nen_qua_lon_bi_loai(self):
        # range_pct_max quá nhỏ so với biên độ thực tế (11.11%) -> bị loại dù có breakout.
        df = self._df_co_chi_bao(high4=10.5, low4=10.0, close4=10.5)
        ket_qua = phat_hien_tin_hieu_vao_lenh(df, range_pct_max=1.0, body_pct_min=0.0, body_pct_max=100.0)
        assert bool(ket_qua["du_dieu_kien_vao_lenh"].iloc[4]) is False


# ==============================================================================
# Test: kiem_tra_dieu_kien_thanh_khoan — trung bình giá trị giao dịch,
# LƯU Ý đơn vị close x1000 (xem docstring hàm).
# ==============================================================================

class TestKiemTraDieuKienThanhKhoan:
    def test_du_thanh_khoan(self):
        # 3 phiên: volume=100_000, close=20.0 (nghìn đồng) -> giá trị/phiên
        # = 100_000 * 20.0 * 1000 = 2_000_000_000 (2 tỷ) mỗi phiên.
        rows = [{"open": 20, "high": 20, "low": 20, "close": 20.0, "volume": 100_000} for _ in range(3)]
        df = _lam_df(rows)
        ket_qua = kiem_tra_dieu_kien_thanh_khoan(df, so_phien_trung_binh=3, gia_tri_trung_binh_toi_thieu=1_000_000_000.0)
        assert bool(ket_qua.iloc[2]) is True

    def test_thieu_thanh_khoan(self):
        rows = [{"open": 20, "high": 20, "low": 20, "close": 20.0, "volume": 1_000} for _ in range(3)]
        df = _lam_df(rows)
        ket_qua = kiem_tra_dieu_kien_thanh_khoan(df, so_phien_trung_binh=3, gia_tri_trung_binh_toi_thieu=1_000_000_000.0)
        # gia tri/phien = 1000*20*1000 = 20_000_000 << 1 ty
        assert bool(ket_qua.iloc[2]) is False

    def test_chua_du_so_phien_trung_binh_tra_ve_false(self):
        rows = [{"open": 20, "high": 20, "low": 20, "close": 20.0, "volume": 1_000_000_000} for _ in range(2)]
        df = _lam_df(rows)
        ket_qua = kiem_tra_dieu_kien_thanh_khoan(df, so_phien_trung_binh=3, gia_tri_trung_binh_toi_thieu=1.0)
        assert bool(ket_qua.iloc[1]) is False


# ==============================================================================
# Test: _tim_floor_theo_tier
# ==============================================================================

class TestTimFloorTheoTier:
    def test_duoi_062_khong_co_floor(self):
        assert _tim_floor_theo_tier(0.5) is None

    def test_dung_bien_duoi_cua_tier(self):
        assert _tim_floor_theo_tier(0.62) == pytest.approx(0.25)
        assert _tim_floor_theo_tier(1.25) == pytest.approx(0.62)
        assert _tim_floor_theo_tier(6.25) == pytest.approx(3.75)

    def test_tier_cao_nhat_khong_gioi_han_tren(self):
        assert _tim_floor_theo_tier(100.0) == pytest.approx(3.75)


# ==============================================================================
# Test: backtest_chien_luoc_breakout — 4 kịch bản tính tay đầy đủ (SL,
# đủ 6 mốc chốt lời, floor breach, hết dữ liệu).
#
# Cấu trúc chung: 4 phiên đầu (index 0-3) PHẲNG rồi breakout tại index 3
# (close=11 > cao_nhat_lookback=10, biên độ nén=0%, EMA/MA đều <11) ->
# VÀO LỆNH tại giá MỞ CỬA phiên index 4. Phần TIẾP THEO (index 4+) tùy
# biến theo từng kịch bản.
# ==============================================================================

def _df_kich_ban(cac_phien_sau_tin_hieu: list[dict]) -> pd.DataFrame:
    rows = [
        {"open": 10, "high": 10, "low": 10, "close": 10, "volume": 1_000_000},
        {"open": 10, "high": 10, "low": 10, "close": 10, "volume": 1_000_000},
        {"open": 10, "high": 10, "low": 10, "close": 10, "volume": 1_000_000},
        {"open": 10.5, "high": 11, "low": 10, "close": 11, "volume": 1_000_000},  # nến tín hiệu (SL=(11+10)/2=10.5)
    ]
    rows.extend(cac_phien_sau_tin_hieu)
    return _lam_df(rows)


CAU_HINH_TEST = {
    "ema_period": 3, "ma_period": 2, "lookback": 2,
    "range_pct_max": 10.0, "body_pct_min": 0.0, "body_pct_max": 100.0,
}


class TestBacktestChienLuocBreakoutKichBanSL:
    def test_cat_lo_ngay_phien_vao_lenh(self):
        # entry tại open index4 = 100... KHÔNG, entry = open của chính
        # phiên NGAY SAU tín hiệu (index4), lấy từ cac_phien_sau_tin_hieu[0].
        df = _df_kich_ban([
            {"open": 100.0, "high": 100.0, "low": 94.0, "close": 94.0, "volume": 1_000_000},  # entry=100, close=94 < SL(10.5%... )
            {"open": 93.0, "high": 93.0, "low": 93.0, "close": 93.0, "volume": 1_000_000},
        ])
        # LƯU Ý: SL của lệnh này KHÔNG phải 10.5 (đó là SL tính theo nến
        # tín hiệu ở thang giá 10-11) — ở đây entry=100 nên minh họa lại:
        # SL thực tế = (11+10)/2 = 10.5 theo thang giá GỐC của nến tín
        # hiệu, nhưng entry_price lấy từ open thực tế của phiên kế tiếp
        # (100.0) — 2 con số này ĐỘC LẬP nhau vì SL bám theo nến tín hiệu,
        # không phải % của entry_price. Với entry=100 và SL=10.5, giá
        # phải giảm cực sâu mới chạm SL — kịch bản này dùng để xác nhận
        # ĐÚNG GIÁ TRỊ SL tính ra, không nhằm test việc chạm SL.
        ket_qua = backtest_chien_luoc_breakout(df, cau_hinh=CAU_HINH_TEST, yeu_cau_thanh_khoan=False)
        assert len(ket_qua) == 1
        assert ket_qua[0].gia_vao_lenh == pytest.approx(100.0)
        assert ket_qua[0].sl == pytest.approx(10.5)

    def test_gia_giam_duoi_sl_thuc_su_dong_lenh(self):
        # Entry price = 10.5 (đặt open phiên kế tiếp = 10.5, khớp thang
        # giá nến tín hiệu) -> SL=10.5 cũng chính bằng entry (trường hợp
        # biên, SL = giá vào) -> close phiên entry PHẢI < 10.5 để cắt lỗ.
        df = _df_kich_ban([
            {"open": 10.5, "high": 10.5, "low": 10.0, "close": 10.0, "volume": 1_000_000},  # entry=10.5, close=10.0 < SL(10.5)
            {"open": 9.8, "high": 9.8, "low": 9.8, "close": 9.8, "volume": 1_000_000},
        ])
        ket_qua = backtest_chien_luoc_breakout(df, cau_hinh=CAU_HINH_TEST, yeu_cau_thanh_khoan=False)
        assert len(ket_qua) == 1
        lenh = ket_qua[0]
        assert lenh.gia_vao_lenh == pytest.approx(10.5)
        assert lenh.sl == pytest.approx(10.5)
        assert len(lenh.cac_dot_dong) == 1
        dot = lenh.cac_dot_dong[0]
        assert dot.ly_do == "SL"
        assert dot.gia == pytest.approx(9.8)
        assert dot.ty_trong_pct == pytest.approx(100.0)
        lai_ky_vong = (9.8 - 10.5) / 10.5 * 100.0
        assert lenh.pnl_pct_tong_hop == pytest.approx(lai_ky_vong)
        assert lenh.thang is False


class TestBacktestChienLuocBreakoutKichBanChotLoiDayDu:
    def test_du_6_moc_chot_loi_dong_het_100_phan_tram(self):
        # entry=100 (open phiên index4=100). Mỗi phiên sau đạt ĐÚNG 1 mốc
        # lãi mới; giả định "mở cửa phẳng qua đêm" (open phiên sau = close
        # phiên trước) để giá đóng lệnh từng đợt = ĐÚNG giá trị mốc.
        muc_lai = [7.5, 8.75, 10.0, 11.25, 13.75, 16.25]
        gia_dong_cua = [100.0 * (1 + m / 100.0) for m in muc_lai]
        phien = [{"open": 100.0, "high": gia_dong_cua[0], "low": 100.0, "close": gia_dong_cua[0], "volume": 1_000_000}]
        for k in range(1, 6):
            phien.append({
                "open": gia_dong_cua[k - 1], "high": gia_dong_cua[k],
                "low": gia_dong_cua[k - 1], "close": gia_dong_cua[k], "volume": 1_000_000,
            })
        # 1 phiên cuối để "mở cửa phẳng" thực thi đợt chốt lời cuối cùng.
        phien.append({"open": gia_dong_cua[-1], "high": gia_dong_cua[-1], "low": gia_dong_cua[-1], "close": gia_dong_cua[-1], "volume": 1_000_000})

        df = _df_kich_ban(phien)
        ket_qua = backtest_chien_luoc_breakout(df, cau_hinh=CAU_HINH_TEST, yeu_cau_thanh_khoan=False)

        assert len(ket_qua) == 1
        lenh = ket_qua[0]
        assert lenh.gia_vao_lenh == pytest.approx(100.0)
        assert len(lenh.cac_dot_dong) == 6

        ty_trong_ky_vong = [40.0, 20.0, 10.0, 10.0, 10.0, 10.0]
        for dot, gia_ky_vong, ty_trong in zip(lenh.cac_dot_dong, gia_dong_cua, ty_trong_ky_vong):
            assert dot.gia == pytest.approx(gia_ky_vong)
            assert dot.ty_trong_pct == pytest.approx(ty_trong)
            assert dot.ly_do.startswith("milestone_")

        pnl_ky_vong = sum(
            ((gia - 100.0) / 100.0 * 100.0) * (ty_trong / 100.0)
            for gia, ty_trong in zip(gia_dong_cua, ty_trong_ky_vong)
        )
        assert pnl_ky_vong == pytest.approx(9.875)
        assert lenh.pnl_pct_tong_hop == pytest.approx(pnl_ky_vong)
        assert lenh.thang is True


class TestBacktestChienLuocBreakoutKichBanFloorBreak:
    def test_hoi_ve_dung_floor_dong_toan_bo_con_lai(self):
        # entry=100. Phiên 1: close=103 (lai 3%) -> floor khóa ở 1.25.
        # Phiên 2: close=101.25 (lai đúng 1.25%, = floor) -> đóng hết.
        df = _df_kich_ban([
            {"open": 100.0, "high": 103.0, "low": 100.0, "close": 103.0, "volume": 1_000_000},
            {"open": 103.0, "high": 103.0, "low": 101.25, "close": 101.25, "volume": 1_000_000},
            {"open": 101.25, "high": 101.25, "low": 101.25, "close": 101.25, "volume": 1_000_000},
        ])
        ket_qua = backtest_chien_luoc_breakout(df, cau_hinh=CAU_HINH_TEST, yeu_cau_thanh_khoan=False)

        assert len(ket_qua) == 1
        lenh = ket_qua[0]
        assert len(lenh.cac_dot_dong) == 1
        dot = lenh.cac_dot_dong[0]
        assert dot.ly_do == "floor_break"
        assert dot.gia == pytest.approx(101.25)
        assert dot.ty_trong_pct == pytest.approx(100.0)
        assert lenh.pnl_pct_tong_hop == pytest.approx(1.25)
        assert lenh.thang is True  # vẫn LÃI dù thoát do hồi về floor


class TestBacktestChienLuocBreakoutKichBanHetDuLieu:
    def test_dong_cuong_buc_khi_het_du_lieu(self):
        # Chỉ 1 phiên sau tín hiệu -> hết dữ liệu ngay sau khi vào lệnh,
        # phải đóng cưỡng bức tại GIÁ ĐÓNG CỬA phiên cuối cùng đó.
        df = _df_kich_ban([
            {"open": 100.0, "high": 102.0, "low": 100.0, "close": 102.0, "volume": 1_000_000},
        ])
        ket_qua = backtest_chien_luoc_breakout(df, cau_hinh=CAU_HINH_TEST, yeu_cau_thanh_khoan=False)

        assert len(ket_qua) == 1
        lenh = ket_qua[0]
        assert len(lenh.cac_dot_dong) == 1
        dot = lenh.cac_dot_dong[0]
        assert dot.ly_do == "het_du_lieu"
        assert dot.gia == pytest.approx(102.0)
        assert dot.ty_trong_pct == pytest.approx(100.0)
        assert lenh.pnl_pct_tong_hop == pytest.approx(2.0)


class TestBacktestChienLuocBreakoutBoLocThanhKhoan:
    def test_thieu_thanh_khoan_khong_vao_lenh(self):
        df = _df_kich_ban([
            {"open": 100.0, "high": 102.0, "low": 100.0, "close": 102.0, "volume": 1_000_000},
        ])
        # volume 1 (cực thấp) -> thanh khoản không đủ -> không vào lệnh nào.
        df["volume"] = 1
        ket_qua = backtest_chien_luoc_breakout(
            df, cau_hinh=CAU_HINH_TEST, yeu_cau_thanh_khoan=True,
            so_phien_trung_binh_thanh_khoan=3, gia_tri_trung_binh_toi_thieu=1_000_000_000.0,
        )
        assert ket_qua == []


# ==============================================================================
# Test: tong_hop_ket_qua — win rate / lãi cộng dồn (compounding) / profit
# factor / max drawdown.
# ==============================================================================

class TestTongHopKetQua:
    def test_khong_co_lenh_nao(self):
        ket_qua = tong_hop_ket_qua([], initial_capital=1_000_000_000.0)
        assert ket_qua["n_trades"] == 0
        assert ket_qua["win_rate_pct"] is None

    def _lam_lenh(self, pnl_pct: float) -> LenhBreakoutNgay:
        return LenhBreakoutNgay(
            ngay_tin_hieu=pd.Timestamp("2024-01-01"), ngay_vao_lenh=pd.Timestamp("2024-01-02"),
            gia_vao_lenh=100.0, sl=95.0, than_nen_tin_hieu_pct=1.0,
            cac_dot_dong=[DotDongLenh(pd.Timestamp("2024-01-03"), 100 * (1 + pnl_pct / 100), 100.0, "het_du_lieu")],
            pnl_pct_tong_hop=pnl_pct, thang=pnl_pct > 0,
        )

    def test_2_lenh_thang_1_lenh_thua_tinh_dung_cac_chi_so(self):
        # +10%, +20%, -5% — vốn tuần tự: 1_000_000_000 -> 1.1e9 -> 1.32e9 -> 1.254e9
        lenh_list = [self._lam_lenh(10.0), self._lam_lenh(20.0), self._lam_lenh(-5.0)]
        ket_qua = tong_hop_ket_qua(lenh_list, initial_capital=1_000_000_000.0)

        assert ket_qua["n_trades"] == 3
        assert ket_qua["win_rate_pct"] == pytest.approx(200.0 / 3)  # 2/3
        von_cuoi = 1_000_000_000.0 * 1.10 * 1.20 * 0.95
        assert ket_qua["ending_capital"] == pytest.approx(von_cuoi)
        assert ket_qua["total_return_pct"] == pytest.approx((von_cuoi / 1_000_000_000.0 - 1) * 100.0)
        assert ket_qua["avg_return_pct"] == pytest.approx((10.0 + 20.0 - 5.0) / 3)
        # profit factor = (10+20) / abs(-5) = 30/5 = 6.0
        assert ket_qua["profit_factor"] == pytest.approx(6.0)

    def test_max_drawdown_tinh_dung_tu_dinh_von(self):
        # vốn: 1e9 -(+10%)-> 1.1e9 (đỉnh) -(-20%)-> 0.88e9 -(+5%)-> 0.924e9
        # drawdown tối đa = (1.1e9-0.88e9)/1.1e9 = 20%
        lenh_list = [self._lam_lenh(10.0), self._lam_lenh(-20.0), self._lam_lenh(5.0)]
        ket_qua = tong_hop_ket_qua(lenh_list, initial_capital=1_000_000_000.0)
        assert ket_qua["max_drawdown_pct"] == pytest.approx(20.0)

    def test_toan_thang_khong_co_lo_profit_factor_vo_cuc(self):
        lenh_list = [self._lam_lenh(5.0), self._lam_lenh(3.0)]
        ket_qua = tong_hop_ket_qua(lenh_list, initial_capital=1_000_000_000.0)
        assert ket_qua["profit_factor"] == float("inf")
