"""
tests/test_stock_character_classifier.py

Unit test cho core/stock_character_classifier.py — mọi giá trị kỳ vọng
đã được DÒ SỐ LIỆU THẬT (chạy trực tiếp hàm) trước khi viết assertion,
theo đúng quy ước rà soát của dự án (xem CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.stock_character_classifier import (
    DiemTinhCach,
    InsufficientDataError,
    NHAN_BUNG_NO_NGAN,
    NHAN_DUT_KHOAT_GIAM,
    NHAN_DUT_KHOAT_TANG,
    NHAN_LINH_XINH,
    NHAN_TRUNG_TINH,
    chuan_hoa_theo_lich_su,
    gan_nhan_tinh_cach,
    gioi_han_ty_trong_theo_tinh_cach,
    he_so_chiet_khau_do_tin_cay,
    kiem_tra_squat_va_churning,
    nhan_dien_churning,
    nhan_dien_squat,
    phan_loai_tinh_cach_co_phieu,
    tinh_choppiness_index,
    tinh_closing_strength,
    tinh_streak_hien_tai,
    tinh_ty_le_dao_chieu,
    tinh_velocity,
)


def _make_df(closes, opens=None, highs=None, lows=None, volumes=None, n=None):
    n = n or len(closes)
    return pd.DataFrame({
        "open": opens or closes,
        "high": highs or [c + 0.5 for c in closes],
        "low": lows or [c - 0.5 for c in closes],
        "close": closes,
        "volume": volumes or [1000] * n,
    })


# ==============================================================================
# Test: tinh_streak_hien_tai
# ==============================================================================

class TestTinhStreakHienTai:
    def test_streak_am_khi_giam_o_cuoi(self):
        df = _make_df([100, 101, 102, 103, 102])  # tăng,tăng,tăng,giảm
        assert tinh_streak_hien_tai(df) == -1

    def test_streak_duong_khi_tang_lien_tiep(self):
        df = _make_df([100, 101, 102, 103, 104])
        assert tinh_streak_hien_tai(df) == 4


# ==============================================================================
# Test: tinh_velocity
# ==============================================================================

class TestTinhVelocity:
    def test_known_velocity_and_pct_change(self):
        closes = [100.0] + list(np.linspace(100, 110, 11))[1:]
        df = _make_df(closes)
        velocity, pct = tinh_velocity(df, n_phien=10)
        assert velocity == pytest.approx(1.0, abs=0.01)
        assert pct == pytest.approx(10.0, abs=0.01)

    def test_raises_for_insufficient_data(self):
        df = _make_df([100, 101, 102])
        with pytest.raises(InsufficientDataError):
            tinh_velocity(df, n_phien=10)


# ==============================================================================
# Test: tinh_choppiness_index
# ==============================================================================

class TestTinhChoppinessIndex:
    def test_low_chop_for_clear_trend(self):
        n = 30
        closes = list(np.linspace(100, 130, n))
        df = _make_df(closes, highs=[c + 0.5 for c in closes], lows=[c - 0.5 for c in closes])
        chop = tinh_choppiness_index(df, n=14).iloc[-1]
        assert chop < 50  # xu hướng rõ ràng -> CHOP thấp

    def test_high_chop_for_sideways_oscillation(self):
        n = 30
        closes = 100 + np.sin(np.arange(n)) * 3
        df = _make_df(list(closes), highs=[c + 2 for c in closes], lows=[c - 2 for c in closes])
        chop = tinh_choppiness_index(df, n=14).iloc[-1]
        assert chop > 50  # đi ngang dao động -> CHOP cao


# ==============================================================================
# Test: tinh_closing_strength
# ==============================================================================

class TestTinhClosingStrength:
    def test_close_at_high_yields_1(self):
        row = pd.Series({"high": 110, "low": 100, "close": 110})
        assert tinh_closing_strength(row) == pytest.approx(1.0)

    def test_close_at_low_yields_0(self):
        row = pd.Series({"high": 110, "low": 100, "close": 100})
        assert tinh_closing_strength(row) == pytest.approx(0.0)

    def test_zero_range_yields_neutral_half(self):
        row = pd.Series({"high": 100, "low": 100, "close": 100})
        assert tinh_closing_strength(row) == pytest.approx(0.5)


# ==============================================================================
# Test: nhan_dien_squat / nhan_dien_churning
# ==============================================================================

class TestNhanDienSquatChurning:
    def test_squat_detected_when_weak_close_above_pivot(self):
        row = pd.Series({"high": 105, "low": 95, "close": 96, "open": 100})
        assert nhan_dien_squat(row, gia_pivot=104) is True

    def test_no_squat_when_strong_close(self):
        row = pd.Series({"high": 105, "low": 95, "close": 104, "open": 100})
        assert nhan_dien_squat(row, gia_pivot=104) is False

    def test_churning_detected_when_high_volume_narrow_range(self):
        row = pd.Series({"high": 100.5, "low": 100.0, "volume": 2000})
        assert nhan_dien_churning(row, volume_ma20=1000, bien_do_pct_percentile_30=1.0) is True

    def test_no_churning_when_volume_normal(self):
        row = pd.Series({"high": 100.5, "low": 100.0, "volume": 900})
        assert nhan_dien_churning(row, volume_ma20=1000, bien_do_pct_percentile_30=1.0) is False


# ==============================================================================
# Test: gan_nhan_tinh_cach
# ==============================================================================

class TestGanNhanTinhCach:
    def test_dut_khoat_tang(self):
        diem = DiemTinhCach(character_score=1.5, choppiness_score=0.5, closing_strength_avg=0.8)
        assert gan_nhan_tinh_cach(diem, velocity_percentile=50, streak=5) == NHAN_DUT_KHOAT_TANG

    def test_dut_khoat_giam(self):
        diem = DiemTinhCach(character_score=-1.5, choppiness_score=0.5, closing_strength_avg=0.2)
        assert gan_nhan_tinh_cach(diem, velocity_percentile=50, streak=-5) == NHAN_DUT_KHOAT_GIAM

    def test_linh_xinh(self):
        diem = DiemTinhCach(character_score=0.0, choppiness_score=1.5, closing_strength_avg=0.5)
        assert gan_nhan_tinh_cach(diem, velocity_percentile=50, streak=1) == NHAN_LINH_XINH

    def test_bung_no_ngan_uu_tien_hon_ca(self):
        # velocity_percentile cao + streak ngắn -> BUNG_NO_NGAN, bất kể character_score
        diem = DiemTinhCach(character_score=0.1, choppiness_score=0.3, closing_strength_avg=0.5)
        assert gan_nhan_tinh_cach(diem, velocity_percentile=90, streak=1) == NHAN_BUNG_NO_NGAN

    def test_trung_tinh_mac_dinh(self):
        diem = DiemTinhCach(character_score=0.1, choppiness_score=0.3, closing_strength_avg=0.5)
        assert gan_nhan_tinh_cach(diem, velocity_percentile=50, streak=1) == NHAN_TRUNG_TINH


# ==============================================================================
# Test: he_so_chiet_khau_do_tin_cay / gioi_han_ty_trong_theo_tinh_cach
# ==============================================================================

class TestTienIchTichHop:
    def test_chiet_khau_khi_choppiness_cao(self):
        assert he_so_chiet_khau_do_tin_cay(1.5) == pytest.approx(0.7)

    def test_khong_chiet_khau_khi_choppiness_thap(self):
        assert he_so_chiet_khau_do_tin_cay(0.5) == pytest.approx(1.0)

    def test_gioi_han_ty_trong_khi_bung_no_ngan(self):
        assert gioi_han_ty_trong_theo_tinh_cach(NHAN_BUNG_NO_NGAN, [], 0.2) == pytest.approx(0.1)

    def test_khong_gioi_han_khi_binh_thuong(self):
        assert gioi_han_ty_trong_theo_tinh_cach(NHAN_TRUNG_TINH, [], 0.2) == pytest.approx(0.2)

    def test_gioi_han_ty_trong_khi_co_churning(self):
        assert gioi_han_ty_trong_theo_tinh_cach(
            NHAN_TRUNG_TINH, ["CHURNING xxx"], 0.2
        ) == pytest.approx(0.1)


# ==============================================================================
# Test: phan_loai_tinh_cach_co_phieu — hàm chính (end-to-end)
# ==============================================================================

class TestPhanLoaiTinhCachCoPhieu:
    def test_raises_for_insufficient_rows(self):
        df = _make_df([100] * 5)
        with pytest.raises(InsufficientDataError):
            phan_loai_tinh_cach_co_phieu("X", df)

    def test_raises_for_missing_columns(self):
        df = pd.DataFrame({"close": [100] * 40})
        with pytest.raises(ValueError):
            phan_loai_tinh_cach_co_phieu("X", df)

    def test_flags_low_confidence_when_history_below_minimum(self):
        n = 60  # dưới MIN_HISTORY_FOR_FULL_CONFIDENCE (500)
        closes = list(np.linspace(100, 130, n))
        df = _make_df(closes, highs=[c + 0.5 for c in closes], lows=[c - 0.5 for c in closes])
        kq = phan_loai_tinh_cach_co_phieu("TEST", df)
        assert kq["do_tin_cay_thap"] is True

    def test_output_structure_matches_spec(self):
        n = 60
        closes = list(np.linspace(100, 130, n))
        df = _make_df(closes, highs=[c + 0.5 for c in closes], lows=[c - 0.5 for c in closes])
        kq = phan_loai_tinh_cach_co_phieu("TEST", df)

        expected_keys = {
            "ma", "ngay_danh_gia", "nhan_tinh_cach", "chi_tiet",
            "character_score", "choppiness_score", "canh_bao",
            "khuyen_nghi_chien_luoc", "do_tin_cay_thap",
        }
        assert expected_keys.issubset(kq.keys())
        assert kq["ma"] == "TEST"
        assert kq["nhan_tinh_cach"] in {
            NHAN_DUT_KHOAT_TANG, NHAN_DUT_KHOAT_GIAM, NHAN_BUNG_NO_NGAN,
            NHAN_LINH_XINH, NHAN_TRUNG_TINH,
        }

    def test_sideways_choppy_data_yields_linh_xinh_or_trung_tinh(self):
        n = 60
        rng = np.random.default_rng(7)
        closes = 100 + np.sin(np.arange(n) * 0.8) * 5 + rng.normal(0, 0.5, n)
        df = _make_df(list(closes), highs=[c + 1.5 for c in closes], lows=[c - 1.5 for c in closes])
        kq = phan_loai_tinh_cach_co_phieu("TEST", df)
        # Dữ liệu dao động lên xuống liên tục (không xu hướng rõ) -> KHÔNG
        # được gán nhãn DỨT KHOÁT (tăng hoặc giảm)
        assert kq["nhan_tinh_cach"] not in {NHAN_DUT_KHOAT_TANG, NHAN_DUT_KHOAT_GIAM}
