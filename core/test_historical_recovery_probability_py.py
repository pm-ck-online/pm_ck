"""
Unit test cho core/historical_recovery_probability.py

Các giá trị kỳ vọng cụ thể đã được DÒ SỐ LIỆU THẬT trước khi viết test
(chạy trực tiếp hàm với dữ liệu giả lập, xác nhận kết quả rồi mới cố định
làm assertion) — tránh việc test "lý thuyết" sai vì hiểu nhầm công thức.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.historical_recovery_probability import (
    DIEU_KIEN_MAC_DINH,
    InvalidRecoveryProbabilityError,
    gop_cum_diem_moc,
    tim_cac_diem_moc_tinh_huong_tuong_tu,
    tinh_do_tap_trung_phuc_hoi,
    tinh_ty_le_phuc_hoi,
    tinh_xac_suat_phuc_hoi_lich_su,
)


def _make_crash_df(n: int = 100, recovery_pct: float = 2.0) -> pd.DataFrame:
    """Tạo DataFrame OHLCV giả lập có ĐÚNG 1 tình huống thỏa mãn bộ điều
    kiện mặc định: giảm ~9.2% trong 3 phiên (50->52), RSI tụt xuống <30,
    volume phiên 52 đột biến gấp ~3 lần TB20, đóng cửa yếu (closing
    strength thấp) tại phiên 52, streak giảm liên tiếp = -3.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    closes = [100.0]
    for i in range(1, n):
        if 50 <= i <= 52:
            closes.append(closes[-1] * (1 - [0.03, 0.03, 0.035][i - 50]))
        else:
            closes.append(closes[-1] * (1 + np.random.normal(0.001, 0.005)))
    closes = np.array(closes)

    opens = closes * (1 + np.random.normal(0, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.003, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.003, n)))
    volumes = np.random.randint(1000, 2000, n).astype(float)

    volumes[52] = 5000
    highs[52] = closes[52] * 1.06   # tạo đóng cửa YẾU (close gần low, xa high)
    lows[52] = closes[52] * 0.995

    if recovery_pct is not None:
        closes[53] = closes[52] * (1 + recovery_pct / 100)

    return pd.DataFrame({
        "date": dates, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


# ==============================================================================
# Test: validate input
# ==============================================================================

class TestValidateInput:
    def test_raises_on_empty_df(self):
        with pytest.raises(InvalidRecoveryProbabilityError):
            tim_cac_diem_moc_tinh_huong_tuong_tu(pd.DataFrame(), DIEU_KIEN_MAC_DINH)

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"date": [1, 2], "close": [1.0, 2.0]})
        with pytest.raises(InvalidRecoveryProbabilityError):
            tim_cac_diem_moc_tinh_huong_tuong_tu(df, DIEU_KIEN_MAC_DINH)


# ==============================================================================
# Test: tim_cac_diem_moc_tinh_huong_tuong_tu
# ==============================================================================

class TestTimCacDiemMoc:
    def test_detects_the_engineered_crash_anchor_day(self):
        df = _make_crash_df()
        points = tim_cac_diem_moc_tinh_huong_tuong_tu(df, DIEU_KIEN_MAC_DINH)
        assert points == [52]

    def test_no_false_positive_on_flat_random_walk(self):
        # Chuỗi biến động nhẹ, không có đợt giảm mạnh nào -> không phát hiện gì.
        np.random.seed(1)
        n = 80
        dates = pd.date_range("2021-01-01", periods=n, freq="D")
        closes = 100 + np.cumsum(np.random.normal(0, 0.3, n))
        df = pd.DataFrame({
            "date": dates,
            "open": closes, "high": closes * 1.005, "low": closes * 0.995,
            "close": closes, "volume": np.random.randint(1000, 1500, n).astype(float),
        })
        points = tim_cac_diem_moc_tinh_huong_tuong_tu(df, DIEU_KIEN_MAC_DINH)
        assert points == []


# ==============================================================================
# Test: gop_cum_diem_moc
# ==============================================================================

class TestGopCumDiemMoc:
    def test_merges_nearby_points_keeps_one_representative(self):
        df = _make_crash_df()
        # 52 và 53 giả lập là 2 "phiên mốc" liền kề (khoảng cách 1 <= 3) ->
        # phải gộp thành 1 đại diện; 90 cách xa (37 phiên) -> giữ riêng.
        clustered = gop_cum_diem_moc(df, [52, 53, 90])
        assert clustered == [52, 90]

    def test_empty_input_returns_empty(self):
        df = _make_crash_df()
        assert gop_cum_diem_moc(df, []) == []

    def test_far_apart_points_stay_separate(self):
        df = _make_crash_df()
        clustered = gop_cum_diem_moc(df, [10, 50, 90])
        assert clustered == [10, 50, 90]


# ==============================================================================
# Test: tinh_ty_le_phuc_hoi
# ==============================================================================

class TestTinhTyLePhucHoi:
    def test_computes_recovery_rate_correctly(self):
        df = _make_crash_df(recovery_pct=2.0)
        result = tinh_ty_le_phuc_hoi(df, [52], so_phien_du_bao=1)
        assert result["so_lan_quan_sat"] == 1
        assert result["so_lan_phuc_hoi"] == 1
        assert result["ty_le_phuc_hoi_pct"] == 100.0
        assert result["pct_thay_doi_trung_binh"] == pytest.approx(2.0, abs=0.01)

    def test_no_recovery_below_threshold(self):
        # Không có phục hồi (giá giảm tiếp) -> ty_le = 0%.
        df = _make_crash_df(recovery_pct=-2.0)
        result = tinh_ty_le_phuc_hoi(df, [52], so_phien_du_bao=1)
        assert result["so_lan_phuc_hoi"] == 0
        assert result["ty_le_phuc_hoi_pct"] == 0.0

    def test_returns_none_rate_when_no_valid_positions(self):
        df = _make_crash_df()
        result = tinh_ty_le_phuc_hoi(df, [], so_phien_du_bao=1)
        assert result["so_lan_quan_sat"] == 0
        assert result["ty_le_phuc_hoi_pct"] is None

    def test_skips_positions_too_close_to_end_of_data(self):
        df = _make_crash_df(n=60)
        # vị trí 58 không đủ 5 phiên dữ liệu tương lai trong df dài 60 phiên.
        result = tinh_ty_le_phuc_hoi(df, [58], so_phien_du_bao=5)
        assert result["so_lan_quan_sat"] == 0


# ==============================================================================
# Test: tinh_xac_suat_phuc_hoi_lich_su (hàm chính, end-to-end)
# ==============================================================================

class TestTinhXacSuatPhucHoiLichSu:
    def test_end_to_end_matches_manual_pipeline(self):
        df = _make_crash_df(recovery_pct=2.0)
        full = tinh_xac_suat_phuc_hoi_lich_su("TEST", df)

        assert full["ma"] == "TEST"
        assert full["so_lan_quan_sat_lich_su"] == 1
        assert full["tong_so_phien_du_lieu_dau_vao"] == 100
        assert set(full["ket_qua_theo_so_phien_du_bao"].keys()) == {
            "sau_1_phien", "sau_3_phien", "sau_5_phien",
        }
        assert full["ket_qua_theo_so_phien_du_bao"]["sau_1_phien"]["ty_le_phuc_hoi_pct"] == 100.0
        assert "canh_bao_phap_ly" in full and "TẦN SUẤT THỰC NGHIỆM" in full["canh_bao_phap_ly"]

    def test_do_tin_cay_rat_thap_khi_co_mau_qua_nho(self):
        # Chỉ 1 lần quan sát -> RAT_THAP theo đúng bảng phân loại mục 5.
        df = _make_crash_df()
        full = tinh_xac_suat_phuc_hoi_lich_su("TEST", df)
        assert full["do_tin_cay_thong_ke"] == "RAT_THAP"

    def test_do_tin_cay_gioi_han_trung_binh_khi_lich_su_ngan(self):
        # Giả lập cỡ mẫu đủ lớn (>=30) nhưng TỔNG độ dài lịch sử vẫn ngắn
        # hơn HISTORY_RECOMMENDED_MIN_SESSIONS -> do_tin_cay bị giới hạn
        # lại KHÔNG được vượt quá TRUNG_BINH (mục 8.2 tài liệu gốc).
        from core.historical_recovery_probability import _xep_do_tin_cay

        do_tin_cay, ghi_chu = _xep_do_tin_cay(so_lan_quan_sat=40, tong_so_phien_lich_su=200)
        assert do_tin_cay == "TRUNG_BINH"
        assert "ít hơn mức khuyến nghị" in ghi_chu

    def test_do_tin_cay_kha_cao_khi_du_lieu_va_mau_du_lon(self):
        from core.historical_recovery_probability import _xep_do_tin_cay

        do_tin_cay, _ = _xep_do_tin_cay(so_lan_quan_sat=40, tong_so_phien_lich_su=1000)
        assert do_tin_cay == "KHA_CAO"

    def test_dieu_kien_loc_tuy_chinh_duoc_ghi_de(self):
        df = _make_crash_df()
        full = tinh_xac_suat_phuc_hoi_lich_su(
            "TEST", df, dieu_kien_loc={"giam_toi_thieu_pct": 50.0}
        )
        # Ngưỡng giảm 50% quá khắt khe -> không mã nào thỏa -> 0 quan sát.
        assert full["so_lan_quan_sat_lich_su"] == 0

    def test_raises_on_invalid_input(self):
        with pytest.raises(InvalidRecoveryProbabilityError):
            tinh_xac_suat_phuc_hoi_lich_su("TEST", pd.DataFrame())


# ==============================================================================
# Test: tinh_do_tap_trung_phuc_hoi (mở rộng cấp thị trường)
# ==============================================================================

class TestTinhDoTapTrungPhucHoi:
    def test_tap_trung_cao_khi_vai_ma_chiem_da_so(self):
        result = tinh_do_tap_trung_phuc_hoi({"VIC": 9.08, "VHM": 6.98, "OTHER": 7.5})
        assert result["ty_le_dong_gop_top3_pct"] == 100.0
        assert result["nhan_dinh"].startswith("TAP_TRUNG_CAO")

    def test_lan_toa_tot_khi_dong_gop_dan_trai(self):
        result = tinh_do_tap_trung_phuc_hoi({
            "A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0, "F": 1.0, "G": 1.0,
        })
        assert result["ty_le_dong_gop_top3_pct"] == pytest.approx(3 / 7 * 100, abs=0.1)
        assert result["nhan_dinh"].startswith("LAN_TOA_TOT")

    def test_raises_on_empty_input(self):
        with pytest.raises(InvalidRecoveryProbabilityError):
            tinh_do_tap_trung_phuc_hoi({})
