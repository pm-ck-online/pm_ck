"""
Unit test cho core/stock_character_classifier.py

Các giá trị kỳ vọng cụ thể đã được DÒ SỐ LIỆU THẬT trước khi viết test
(chạy trực tiếp hàm với dữ liệu giả lập, xác nhận kết quả rồi mới cố định
làm assertion) — tránh việc test "lý thuyết" sai vì hiểu nhầm công thức.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.stock_character_classifier import (
    NHAN_BUNG_NO_NGAN,
    NHAN_DUT_KHOAT_GIAM,
    NHAN_DUT_KHOAT_TANG,
    NHAN_LINH_XINH,
    NHAN_TRUNG_TINH,
    DiemTinhCach,
    InsufficientDataError,
    chuan_hoa_theo_lich_su,
    clip,
    dem_churning_gan_day,
    gan_nhan_tinh_cach,
    gioi_han_ty_trong_theo_tinh_cach,
    he_so_chiet_khau_do_tin_cay,
    kiem_tra_squat_va_churning,
    nhan_dien_churning,
    nhan_dien_squat,
    phan_loai_tinh_cach_co_phieu,
    tinh_autocorrelation,
    tinh_character_score,
    tinh_choppiness_index,
    tinh_closing_strength,
    tinh_streak_hien_tai,
    tinh_ty_le_dao_chieu,
    tinh_velocity,
)


def _make_df(closes, opens=None, highs=None, lows=None, volumes=None):
    n = len(closes)
    return pd.DataFrame({
        "open": opens or closes,
        "high": highs or [c + 0.5 for c in closes],
        "low": lows or [c - 0.5 for c in closes],
        "close": closes,
        "volume": volumes or [1000] * n,
    })


# ==============================================================================
# Test: clip
# ==============================================================================

class TestClip:
    def test_within_range(self):
        assert clip(1.0, -2, 2) == 1.0

    def test_clips_above(self):
        assert clip(5.0, -2, 2) == 2.0

    def test_clips_below(self):
        assert clip(-5.0, -2, 2) == -2.0


# ==============================================================================
# Test: tinh_streak_hien_tai — đã dò số liệu thật
# ==============================================================================

class TestTinhStreakHienTai:
    def test_known_downtrend_streak(self):
        # Đã dò kiểm: 100->99 giảm 5 phiên liên tiếp -> streak = -5
        closes = [100, 101, 102, 103, 104, 103, 102, 101, 100, 99]
        df = _make_df(closes)
        assert tinh_streak_hien_tai(df) == -5

    def test_known_uptrend_streak(self):
        # Tính tay: dấu return = [-1,-1,+1,+1,+1,+1] -> 4 phiên +1 liên tiếp cuối
        closes = [100, 99, 98, 99, 100, 101, 102]
        df = _make_df(closes)
        assert tinh_streak_hien_tai(df) == 4

    def test_flat_last_session_yields_zero(self):
        closes = [100, 101, 102, 102]  # phiên cuối đứng giá
        df = _make_df(closes)
        assert tinh_streak_hien_tai(df) == 0


# ==============================================================================
# Test: tinh_velocity — đã dò số liệu thật
# ==============================================================================

class TestTinhVelocity:
    def test_known_uptrend_velocity(self):
        # Đã dò kiểm: tăng đều 100->110 qua 10 phiên -> velocity=1.0%/phiên, pct=10.0%
        closes = [100 + i for i in range(11)]
        df = _make_df(closes)
        velocity, pct_change = tinh_velocity(df, n_phien=10)
        assert velocity == pytest.approx(1.0, abs=0.01)
        assert pct_change == pytest.approx(10.0, abs=0.01)

    def test_raises_for_insufficient_data(self):
        df = _make_df([100, 101, 102])
        with pytest.raises(InsufficientDataError):
            tinh_velocity(df, n_phien=10)


# ==============================================================================
# Test: tinh_closing_strength — đã dò số liệu thật
# ==============================================================================

class TestTinhClosingStrength:
    def test_close_at_middle_yields_half(self):
        row = pd.Series({"high": 110, "low": 90, "close": 100})
        assert tinh_closing_strength(row) == pytest.approx(0.5)

    def test_close_near_high_yields_strong(self):
        row = pd.Series({"high": 110, "low": 90, "close": 109})
        assert tinh_closing_strength(row) == pytest.approx(0.95)

    def test_zero_range_yields_half(self):
        row = pd.Series({"high": 100, "low": 100, "close": 100})
        assert tinh_closing_strength(row) == pytest.approx(0.5)


# ==============================================================================
# Test: tinh_ty_le_dao_chieu — đã dò số liệu thật
# ==============================================================================

class TestTinhTyLeDaoChieu:
    def test_known_zigzag_high_reversal(self):
        # Đã dò kiểm: chuỗi zig-zag 31 điểm -> reversal_rate ~0.9667
        closes = [100, 101] * 15 + [100]
        df = _make_df(closes)
        result = tinh_ty_le_dao_chieu(df, n_phien=30)
        assert result == pytest.approx(0.9667, abs=0.001)

    def test_monotonic_trend_yields_zero_reversal(self):
        closes = [100 + i for i in range(31)]
        df = _make_df(closes)
        result = tinh_ty_le_dao_chieu(df, n_phien=30)
        assert result == pytest.approx(0.0)


# ==============================================================================
# Test: tinh_choppiness_index — đã dò số liệu thật
# ==============================================================================

class TestTinhChoppinessIndex:
    def test_choppy_sideways_data_yields_high_chop(self):
        rng = np.random.default_rng(0)
        closes = [100 + rng.normal(0, 1) for _ in range(30)]
        df = _make_df(closes, highs=[c + 2 for c in closes], lows=[c - 2 for c in closes])
        chop = tinh_choppiness_index(df, n=14)
        assert chop.iloc[-1] == pytest.approx(82.88, abs=0.1)

    def test_strong_trend_yields_lower_chop_than_choppy(self):
        # So sánh tương đối: xu hướng dứt khoát phải có CHOP thấp hơn hẳn
        # dữ liệu đi ngang hỗn loạn ở test trên (không cố định số tuyệt đối
        # vì công thức khá nhạy với range/TR, chỉ kiểm tra tính hợp lý).
        closes = [100 + i * 3 for i in range(30)]
        df = _make_df(closes, highs=[c + 1 for c in closes], lows=[c - 1 for c in closes])
        chop_trend = tinh_choppiness_index(df, n=14).iloc[-1]
        assert chop_trend < 82.88


# ==============================================================================
# Test: nhan_dien_squat / nhan_dien_churning
# ==============================================================================

class TestNhanDienSquatChurning:
    def test_squat_detected_when_breakout_but_weak_close(self):
        row = pd.Series({"high": 115, "low": 100, "close": 103})  # closing_strength=0.2, vượt pivot 110
        assert nhan_dien_squat(row, gia_pivot=110) is True

    def test_no_squat_when_strong_close(self):
        row = pd.Series({"high": 115, "low": 100, "close": 113})  # closing_strength cao
        assert nhan_dien_squat(row, gia_pivot=110) is False

    def test_no_squat_when_high_does_not_break_pivot(self):
        row = pd.Series({"high": 105, "low": 100, "close": 101})
        assert nhan_dien_squat(row, gia_pivot=110) is False

    def test_churning_detected_high_volume_narrow_range(self):
        row = pd.Series({"high": 100.5, "low": 100.0, "volume": 200000})
        assert nhan_dien_churning(row, volume_ma20=100000, bien_do_pct_percentile_30=1.0) is True

    def test_no_churning_when_volume_normal(self):
        row = pd.Series({"high": 100.5, "low": 100.0, "volume": 100000})
        assert nhan_dien_churning(row, volume_ma20=100000, bien_do_pct_percentile_30=1.0) is False


# ==============================================================================
# Test: chuan_hoa_theo_lich_su
# ==============================================================================

class TestChuanHoaTheoLichSu:
    def test_median_value_yields_around_50th_percentile(self):
        history = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = chuan_hoa_theo_lich_su(3.0, history)
        assert result == pytest.approx(50.0)

    def test_max_value_yields_90th_percentile_with_mean_kind(self):
        # percentileofscore(kind="mean") = trung bình (weak, strict) percentile
        # -> weak=100 (<=x), strict=80 (<x) -> mean=90, không phải 100
        history = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = chuan_hoa_theo_lich_su(5.0, history)
        assert result == pytest.approx(90.0)

    def test_empty_history_yields_neutral_50(self):
        result = chuan_hoa_theo_lich_su(3.0, np.array([]))
        assert result == 50.0


# ==============================================================================
# Test: gan_nhan_tinh_cach
# ==============================================================================

class TestGanNhanTinhCach:
    def test_high_velocity_percentile_low_streak_yields_bung_no_ngan(self):
        diem = DiemTinhCach(character_score=0.5, choppiness_score=0.5, closing_strength_avg=0.6)
        nhan = gan_nhan_tinh_cach(diem, velocity_percentile=90, streak=2)
        assert nhan == NHAN_BUNG_NO_NGAN

    def test_strong_positive_score_low_chop_yields_dut_khoat_tang(self):
        diem = DiemTinhCach(character_score=1.5, choppiness_score=0.5, closing_strength_avg=0.7)
        nhan = gan_nhan_tinh_cach(diem, velocity_percentile=60, streak=5)
        assert nhan == NHAN_DUT_KHOAT_TANG

    def test_strong_negative_score_low_chop_yields_dut_khoat_giam(self):
        diem = DiemTinhCach(character_score=-1.5, choppiness_score=0.5, closing_strength_avg=0.3)
        nhan = gan_nhan_tinh_cach(diem, velocity_percentile=60, streak=-5)
        assert nhan == NHAN_DUT_KHOAT_GIAM

    def test_high_chop_yields_linh_xinh(self):
        diem = DiemTinhCach(character_score=0.2, choppiness_score=1.5, closing_strength_avg=0.5)
        nhan = gan_nhan_tinh_cach(diem, velocity_percentile=40, streak=1)
        assert nhan == NHAN_LINH_XINH

    def test_default_neutral(self):
        diem = DiemTinhCach(character_score=0.2, choppiness_score=0.5, closing_strength_avg=0.5)
        nhan = gan_nhan_tinh_cach(diem, velocity_percentile=40, streak=1)
        assert nhan == NHAN_TRUNG_TINH


# ==============================================================================
# Test: he_so_chiet_khau_do_tin_cay / gioi_han_ty_trong_theo_tinh_cach
# ==============================================================================

class TestTienIchTichHop:
    def test_discount_applied_when_choppiness_above_threshold(self):
        assert he_so_chiet_khau_do_tin_cay(1.4) == 0.7

    def test_no_discount_when_choppiness_at_or_below_threshold(self):
        assert he_so_chiet_khau_do_tin_cay(1.0) == 1.0
        assert he_so_chiet_khau_do_tin_cay(0.5) == 1.0

    def test_gioi_han_ty_trong_halves_for_bung_no_ngan(self):
        result = gioi_han_ty_trong_theo_tinh_cach(NHAN_BUNG_NO_NGAN, [], 0.20)
        assert result == pytest.approx(0.10)

    def test_gioi_han_ty_trong_halves_for_churning_warning(self):
        result = gioi_han_ty_trong_theo_tinh_cach(
            NHAN_TRUNG_TINH, ["CHURNING — nghi ngờ phân phối"], 0.20
        )
        assert result == pytest.approx(0.10)

    def test_gioi_han_ty_trong_unchanged_when_normal(self):
        result = gioi_han_ty_trong_theo_tinh_cach(NHAN_DUT_KHOAT_TANG, [], 0.20)
        assert result == pytest.approx(0.20)


# ==============================================================================
# Test: phan_loai_tinh_cach_co_phieu — hàm chính, end-to-end
# ==============================================================================

def _make_realistic_df(n=550, seed=1, trend=0.0):
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(trend, 1.0, n))
    df = pd.DataFrame({
        "close": closes,
        "open": np.concatenate([[closes[0]], closes[:-1]]),
        "high": closes + np.abs(rng.normal(0.5, 0.3, n)),
        "low": closes - np.abs(rng.normal(0.5, 0.3, n)),
        "volume": rng.integers(500_000, 1_500_000, n),
    })
    return df


class TestPhanLoaiTinhCachCoPhieu:
    def test_raises_for_missing_columns(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        with pytest.raises(ValueError):
            phan_loai_tinh_cach_co_phieu("HPG", df)

    def test_raises_for_insufficient_rows(self):
        df = _make_realistic_df(n=10)
        with pytest.raises(InsufficientDataError):
            phan_loai_tinh_cach_co_phieu("HPG", df)

    def test_output_structure_matches_spec(self):
        df = _make_realistic_df(n=550)
        result = phan_loai_tinh_cach_co_phieu("HPG", df)

        expected_keys = {
            "ma", "ngay_danh_gia", "nhan_tinh_cach", "chi_tiet",
            "character_score", "choppiness_score", "canh_bao",
            "khuyen_nghi_chien_luoc", "do_tin_cay_thap",
        }
        assert expected_keys.issubset(result.keys())
        assert result["ma"] == "HPG"
        assert result["nhan_tinh_cach"] in {
            NHAN_DUT_KHOAT_TANG, NHAN_DUT_KHOAT_GIAM, NHAN_BUNG_NO_NGAN,
            NHAN_LINH_XINH, NHAN_TRUNG_TINH,
        }

    def test_do_tin_cay_thap_flagged_when_history_short(self):
        df = _make_realistic_df(n=100)  # dưới MIN_HISTORY_FOR_FULL_CONFIDENCE=500
        result = phan_loai_tinh_cach_co_phieu("HPG", df)
        assert result["do_tin_cay_thap"] is True

    def test_do_tin_cay_thap_false_when_enough_history(self):
        df = _make_realistic_df(n=550)
        result = phan_loai_tinh_cach_co_phieu("HPG", df)
        assert result["do_tin_cay_thap"] is False

    def test_strong_uptrend_data_runs_without_error_and_yields_valid_label(self):
        # LƯU Ý: vì percentile tính NỘI TẠI (so với chính lịch sử của mã),
        # dữ liệu random-walk có trend vẫn có thể ra nhãn TRUNG_TINH tùy
        # phân phối cụ thể — không cố định 1 nhãn duy nhất, chỉ kiểm tra
        # chạy không lỗi và nhãn hợp lệ (đã kiểm tra kỹ hơn ở
        # test_output_structure_matches_spec).
        df = _make_realistic_df(n=550, trend=1.5, seed=42)
        result = phan_loai_tinh_cach_co_phieu("HPG", df)
        assert result["nhan_tinh_cach"] in {
            NHAN_DUT_KHOAT_TANG, NHAN_DUT_KHOAT_GIAM, NHAN_BUNG_NO_NGAN,
            NHAN_LINH_XINH, NHAN_TRUNG_TINH,
        }

    def test_chi_tiet_contains_all_expected_fields(self):
        df = _make_realistic_df(n=550)
        result = phan_loai_tinh_cach_co_phieu("HPG", df)
        expected_detail_keys = {
            "streak_hien_tai", "velocity_10_phien_pct", "pct_change_10_phien",
            "choppiness_index", "closing_strength_trung_binh",
            "ty_le_dao_chieu_30_phien", "autocorrelation_lag1",
            "streak_percentile_noi_tai", "velocity_percentile_noi_tai",
        }
        assert expected_detail_keys.issubset(result["chi_tiet"].keys())
