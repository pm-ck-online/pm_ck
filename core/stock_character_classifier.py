"""
core/stock_character_classifier.py

Module phân loại "tính cách giao dịch" (trading character) của từng mã cổ phiếu:
mức độ DỨT KHOÁT/quyết liệt hay LÌNH XÌNH/do dự trong hành vi giá, dựa trên
dữ liệu lịch sử OHLCV.

Đây KHÔNG phải module xác định xu hướng (đã có market_regime_detector.py) —
đây là đặc tính "vận động" nội tại của riêng từng mã, tính theo percentile
NỘI TẠI (so với chính lịch sử của mã đó), không dùng ngưỡng cứng chung cho
mọi mã.

CHỈ ĐỌC DỮ LIỆU / TÍNH TOÁN / PHÂN LOẠI — KHÔNG đặt lệnh, KHÔNG khuyến nghị
đầu tư cá nhân hóa. Xem mục 9 trong prompt gốc.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


# ============================================================================
# Cấu hình mặc định (có thể override khi gọi hàm hoặc qua config.yaml)
# ============================================================================

DEFAULT_LOOKBACK_WINDOW = 20        # số phiên gần nhất để đánh giá "tính cách hiện tại"
DEFAULT_HISTORY_WINDOW = 500        # số phiên nền để tính percentile nội tại (~2 năm)
MIN_HISTORY_FOR_FULL_CONFIDENCE = 500  # dưới ngưỡng này -> gắn cờ do_tin_cay_thap
VELOCITY_N_PHIEN = 10                # cửa sổ tính velocity (khớp ví dụ minh họa: 10% / 10 phiên)
CHOP_N = 14                          # chu kỳ Choppiness Index
REVERSAL_N_PHIEN = 30                # cửa sổ tính tỷ lệ đảo chiều
AUTOCORR_WINDOW = 90                 # cửa sổ tính autocorrelation
AUTOCORR_LAG = 1

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


# ============================================================================
# Exceptions
# ============================================================================

class InsufficientDataError(ValueError):
    """Dữ liệu OHLCV không đủ để tính toán đáng tin cậy."""


def _validate_df(df: pd.DataFrame, min_rows: int = 30) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame thiếu cột bắt buộc: {missing}")
    if len(df) < min_rows:
        raise InsufficientDataError(
            f"Cần tối thiểu {min_rows} phiên, chỉ có {len(df)} phiên."
        )


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================================
# 2.1 — Streak (chuỗi phiên liên tiếp cùng chiều)
# ============================================================================

def tinh_dau_return(df: pd.DataFrame) -> pd.Series:
    """Dấu của return từng phiên: +1 tăng, -1 giảm, 0 đứng giá."""
    return np.sign(df["close"].diff())


def tinh_streak_hien_tai(df: pd.DataFrame) -> int:
    """
    Trả về độ dài streak hiện tại, CÓ DẤU:
      dương = đang trong chuỗi tăng liên tiếp
      âm    = đang trong chuỗi giảm liên tiếp
    """
    dau = tinh_dau_return(df).dropna()
    if len(dau) == 0:
        return 0
    dau_gan_nhat = dau.iloc[-1]
    if dau_gan_nhat == 0:
        return 0
    streak = 1
    for i in range(len(dau) - 2, -1, -1):
        if dau.iloc[i] == dau_gan_nhat:
            streak += 1
        else:
            break
    return int(streak * dau_gan_nhat)


def tinh_chuoi_streak_toan_bo(df: pd.DataFrame) -> np.ndarray:
    """
    Trả về mảng độ dài (giá trị tuyệt đối) của TẤT CẢ các streak đã xảy ra
    trong toàn bộ chuỗi df — dùng làm nền so sánh percentile nội tại.
    """
    dau = tinh_dau_return(df).dropna().to_numpy()
    if len(dau) == 0:
        return np.array([])
    lengths = []
    cur_sign = dau[0]
    cur_len = 1
    for s in dau[1:]:
        if s == cur_sign and s != 0:
            cur_len += 1
        else:
            if cur_sign != 0:
                lengths.append(cur_len)
            cur_sign = s
            cur_len = 1
    if cur_sign != 0:
        lengths.append(cur_len)
    return np.array(lengths, dtype=float)


# ============================================================================
# 2.2 — Velocity (tốc độ biến động, %/phiên)
# ============================================================================

def tinh_velocity(df: pd.DataFrame, n_phien: int = VELOCITY_N_PHIEN) -> tuple[float, float]:
    """
    Trả về (velocity %/phiên có dấu, % thay đổi tổng trong n_phien).
    Ví dụ: giảm 10% trong 10 phiên -> velocity = -1.0 %/phiên.
    """
    if len(df) < n_phien + 1:
        raise InsufficientDataError(f"Cần tối thiểu {n_phien + 1} phiên để tính velocity.")
    gia_dau = df["close"].iloc[-(n_phien + 1)]
    gia_cuoi = df["close"].iloc[-1]
    if gia_dau == 0:
        return 0.0, 0.0
    pct_change = (gia_cuoi - gia_dau) / gia_dau * 100.0
    velocity = pct_change / n_phien
    return float(velocity), float(pct_change)


def tinh_chuoi_velocity_toan_bo(df: pd.DataFrame, n_phien: int = VELOCITY_N_PHIEN) -> np.ndarray:
    """Mảng velocity rolling (%/phiên) trên toàn bộ chuỗi — nền percentile nội tại."""
    pct_change_n = df["close"].pct_change(periods=n_phien) * 100.0
    velocity_series = (pct_change_n / n_phien).dropna()
    return velocity_series.to_numpy()


# ============================================================================
# 2.3 — Choppiness Index (CHOP)
# ============================================================================

def tinh_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def tinh_choppiness_index(df: pd.DataFrame, n: int = CHOP_N) -> pd.Series:
    """
    CHOP -> 100: đi ngang / hỗn loạn (lình xình).
    CHOP -> 0  : xu hướng rất rõ ràng (dứt khoát), bất kể chiều tăng/giảm.
    Trả về cả chuỗi (Series) để có thể lấy .iloc[-1] hoặc dùng làm nền lịch sử.
    """
    tr = tinh_true_range(df)
    sum_tr_n = tr.rolling(n).sum()
    range_n = df["high"].rolling(n).max() - df["low"].rolling(n).min()
    with np.errstate(divide="ignore", invalid="ignore"):
        chop = 100 * np.log10(sum_tr_n / range_n) / np.log10(n)
    return chop.replace([np.inf, -np.inf], np.nan)


# ============================================================================
# 2.4 — Closing Strength + nhận diện Squat / Churning (Minervini)
# ============================================================================

def tinh_closing_strength(row: pd.Series) -> float:
    """
    (close - low) / (high - low).
    > 0.7 : đóng cửa mạnh.  < 0.3 : đóng cửa yếu.
    """
    rng = row["high"] - row["low"]
    if rng == 0:
        return 0.5
    return float((row["close"] - row["low"]) / rng)


def nhan_dien_squat(row: pd.Series, gia_pivot: float) -> bool:
    """Bứt phá trong phiên (high vượt pivot) nhưng đóng cửa yếu -> "giả bứt phá"."""
    return bool(row["high"] > gia_pivot and tinh_closing_strength(row) < 0.4)


def nhan_dien_churning(row: pd.Series, volume_ma20: float, bien_do_pct_percentile_30: float) -> bool:
    """Volume rất lớn nhưng biên độ giá trong ngày rất hẹp -> nghi ngờ phân phối ẩn."""
    if row["low"] == 0 or volume_ma20 == 0 or np.isnan(volume_ma20):
        return False
    bien_do_pct = (row["high"] - row["low"]) / row["low"] * 100.0
    return bool(row["volume"] > 1.5 * volume_ma20 and bien_do_pct < bien_do_pct_percentile_30)


def dem_churning_gan_day(df: pd.DataFrame, n: int = 5, volume_ma_window: int = 20) -> int:
    """Đếm số phiên có dấu hiệu churning trong n phiên gần nhất."""
    if len(df) < volume_ma_window + n:
        return 0
    volume_ma = df["volume"].rolling(volume_ma_window).mean()
    bien_do_pct_series = (df["high"] - df["low"]) / df["low"].replace(0, np.nan) * 100.0
    p30 = np.nanpercentile(bien_do_pct_series.iloc[-DEFAULT_HISTORY_WINDOW:], 30)

    count = 0
    recent = df.iloc[-n:]
    recent_vol_ma = volume_ma.iloc[-n:]
    for (_, row), vma in zip(recent.iterrows(), recent_vol_ma):
        if nhan_dien_churning(row, vma, p30):
            count += 1
    return count


def kiem_tra_squat_va_churning(df: pd.DataFrame, gia_pivot_gan_nhat: Optional[float] = None) -> list[str]:
    canh_bao: list[str] = []
    if len(df) < 30:
        return canh_bao

    row_gan_nhat = df.iloc[-1]
    pivot = gia_pivot_gan_nhat
    if pivot is None:
        # Mặc định: pivot tham khảo = đỉnh cao nhất 20 phiên trước phiên gần nhất
        pivot = df["high"].iloc[-21:-1].max()

    if pd.notna(pivot) and nhan_dien_squat(row_gan_nhat, pivot):
        canh_bao.append(
            "SQUAT — bứt phá trong phiên nhưng đóng cửa yếu, cần xác nhận thêm 1-2 phiên"
        )

    if dem_churning_gan_day(df, n=5) >= 2:
        canh_bao.append(
            "CHURNING — nghi ngờ dòng tiền lớn đang phân phối ẩn trong vài phiên gần đây"
        )

    return canh_bao


# ============================================================================
# 2.5 — Tỷ lệ đảo chiều (Reversal Frequency)
# ============================================================================

def tinh_ty_le_dao_chieu(df: pd.DataFrame, n_phien: int = REVERSAL_N_PHIEN) -> float:
    dau = tinh_dau_return(df).dropna()
    if len(dau) < n_phien:
        n_phien = len(dau)
    if n_phien == 0:
        return 0.0
    dau_recent = dau.iloc[-n_phien:]
    so_lan_doi_chieu = int((dau_recent.diff().abs() > 0).sum())
    return so_lan_doi_chieu / n_phien


# ============================================================================
# 2.6 — Autocorrelation (momentum vs mean-reversion)
# ============================================================================

def tinh_autocorrelation(df: pd.DataFrame, lag: int = AUTOCORR_LAG, window: int = AUTOCORR_WINDOW) -> float:
    returns = df["close"].pct_change().dropna()
    if len(returns) < window:
        window = len(returns)
    if window <= lag + 1:
        return 0.0
    returns_w = returns.iloc[-window:]
    val = returns_w.autocorr(lag=lag)
    return float(val) if pd.notna(val) else 0.0


# ============================================================================
# 3 — Chuẩn hóa theo percentile nội tại
# ============================================================================

def chuan_hoa_theo_lich_su(gia_tri_hien_tai: float, chuoi_lich_su: np.ndarray) -> float:
    """Percentile của giá trị hiện tại so với chính lịch sử của mã đó (0-100)."""
    chuoi_lich_su = chuoi_lich_su[~np.isnan(chuoi_lich_su)]
    if len(chuoi_lich_su) == 0:
        return 50.0  # không có nền lịch sử -> mặc định trung tính
    return float(percentileofscore(chuoi_lich_su, gia_tri_hien_tai, kind="mean"))


# ============================================================================
# 4 — Character Score tổng hợp
# ============================================================================

@dataclass
class DiemTinhCach:
    character_score: float
    choppiness_score: float
    closing_strength_avg: float


def tinh_character_score(
    streak: int,
    velocity: float,
    chop: float,
    closing_strength_avg: float,
    reversal_rate: float,
    autocorr: float,
    streak_percentile: float,
    velocity_percentile: float,
) -> DiemTinhCach:
    if np.isnan(chop):
        chop = 50.0  # trung tính nếu chưa đủ dữ liệu tính CHOP

    # Hướng vận động hiện tại: ưu tiên streak, nếu streak=0 thì dùng dấu velocity.
    # LƯU Ý — đây là điểm đã hiệu chỉnh so với bản nháp đầu: CHOP và autocorrelation
    # (không mang dấu tự nhiên) phải được nhân theo `huong` thay vì cộng trực tiếp,
    # nếu không sẽ triệt tiêu 2 thành phần streak/velocity có dấu và làm sai lệch
    # dấu tổng thể của character_score (đã phát hiện qua unit test end-to-end).
    if streak != 0:
        huong = np.sign(streak)
    elif velocity != 0:
        huong = np.sign(velocity)
    else:
        huong = 0

    do_lon_dut_khoat = (
        0.30 * clip(streak_percentile / 50 - 1, -2, 2)
        + 0.25 * clip(velocity_percentile / 50 - 1, -2, 2)
        + 0.25 * clip((50 - chop) / 25, -2, 2)
        + 0.20 * clip(autocorr / 0.15, -2, 2)
    )
    diem_dut_khoat = huong * clip(do_lon_dut_khoat, -2, 2)

    diem_linh_xinh = (
        0.5 * clip(chop / 61.8, 0, 2)
        + 0.5 * clip(reversal_rate / 0.4, 0, 2)
    )

    return DiemTinhCach(
        character_score=round(float(diem_dut_khoat), 3),
        choppiness_score=round(float(diem_linh_xinh), 3),
        closing_strength_avg=round(float(closing_strength_avg), 3),
    )


# ============================================================================
# 5 — Gán nhãn cuối cùng
# ============================================================================

NHAN_DUT_KHOAT_TANG = "DUT_KHOAT_TANG"
NHAN_DUT_KHOAT_GIAM = "DUT_KHOAT_GIAM"
NHAN_BUNG_NO_NGAN = "BUNG_NO_NGAN"
NHAN_LINH_XINH = "LINH_XINH"
NHAN_TRUNG_TINH = "TRUNG_TINH"


def gan_nhan_tinh_cach(diem: DiemTinhCach, velocity_percentile: float, streak: int) -> str:
    if velocity_percentile >= 85 and abs(streak) < 3:
        return NHAN_BUNG_NO_NGAN
    if diem.character_score >= 1.0 and diem.choppiness_score < 0.8:
        return NHAN_DUT_KHOAT_TANG
    if diem.character_score <= -1.0 and diem.choppiness_score < 0.8:
        return NHAN_DUT_KHOAT_GIAM
    if diem.choppiness_score >= 1.2:
        return NHAN_LINH_XINH
    return NHAN_TRUNG_TINH


def goi_y_chien_luoc_theo_nhan(nhan: str) -> str:
    goi_y = {
        NHAN_DUT_KHOAT_TANG: (
            "Ưu tiên chiến lược theo trend: mua theo Pullback/Breakout "
            "(xem stock_signal_engine.py Mục 2.2)"
        ),
        NHAN_DUT_KHOAT_GIAM: (
            "Tránh bắt đáy sớm; nếu đã nắm giữ, ưu tiên chiến lược Sell the rally "
            "hoặc cắt lỗ theo kế hoạch"
        ),
        NHAN_BUNG_NO_NGAN: (
            "Rủi ro cao — nếu giao dịch, giảm khối lượng vị thế và siết chặt cắt lỗ "
            "hơn mức thông thường"
        ),
        NHAN_LINH_XINH: (
            "Ưu tiên chiến lược giao dịch biên độ (Range trading): mua hỗ trợ/bán "
            "kháng cự, R:R 1:1-1:2"
        ),
        NHAN_TRUNG_TINH: "Chưa đủ tín hiệu rõ ràng — theo dõi thêm, không nên chủ động vào lệnh lớn",
    }
    return goi_y.get(nhan, "Chưa xác định")


# ============================================================================
# 6 — Hàm chính (entry point của module)
# ============================================================================

def phan_loai_tinh_cach_co_phieu(
    ma: str,
    df_ohlcv: pd.DataFrame,
    lookback_window: int = DEFAULT_LOOKBACK_WINDOW,
    history_window: int = DEFAULT_HISTORY_WINDOW,
    gia_pivot_gan_nhat: Optional[float] = None,
) -> dict:
    """
    Hàm chính: phân loại tính cách giao dịch của 1 mã cổ phiếu.

    Parameters
    ----------
    ma : mã cổ phiếu (vd "HPG")
    df_ohlcv : DataFrame có cột open, high, low, close, volume, index là ngày
               tăng dần theo thời gian (phiên cũ nhất ở đầu, mới nhất ở cuối)
    lookback_window : số phiên gần nhất để đánh giá tính cách hiện tại
    history_window : số phiên nền để tính percentile nội tại
    gia_pivot_gan_nhat : mức giá pivot tham khảo để nhận diện Squat (tùy chọn;
               nếu không truyền, tự suy ra từ đỉnh 20 phiên trước phiên gần nhất)

    Returns
    -------
    dict theo đúng cấu trúc output đã đặc tả ở Mục 7 của prompt gốc.
    """
    _validate_df(df_ohlcv, min_rows=max(VELOCITY_N_PHIEN + 1, 30))

    do_tin_cay_thap = len(df_ohlcv) < MIN_HISTORY_FOR_FULL_CONFIDENCE

    df_hist = df_ohlcv.iloc[-history_window:] if len(df_ohlcv) > history_window else df_ohlcv
    n_recent = min(lookback_window, len(df_ohlcv))
    df_recent = df_ohlcv.iloc[-n_recent:]

    streak = tinh_streak_hien_tai(df_ohlcv)
    velocity, pct_change_10p = tinh_velocity(df_ohlcv, n_phien=VELOCITY_N_PHIEN)
    chop_series = tinh_choppiness_index(df_ohlcv, n=CHOP_N)
    chop = float(chop_series.iloc[-1]) if pd.notna(chop_series.iloc[-1]) else float("nan")
    closing_strength_avg = float(df_recent.apply(tinh_closing_strength, axis=1).mean())
    reversal_rate = tinh_ty_le_dao_chieu(df_ohlcv, n_phien=REVERSAL_N_PHIEN)
    autocorr = tinh_autocorrelation(df_ohlcv, lag=AUTOCORR_LAG, window=AUTOCORR_WINDOW)

    chuoi_streak_lich_su = tinh_chuoi_streak_toan_bo(df_hist)
    chuoi_velocity_lich_su = tinh_chuoi_velocity_toan_bo(df_hist, n_phien=VELOCITY_N_PHIEN)

    streak_percentile = chuan_hoa_theo_lich_su(abs(streak), np.abs(chuoi_streak_lich_su))
    velocity_percentile = chuan_hoa_theo_lich_su(abs(velocity), np.abs(chuoi_velocity_lich_su))

    diem = tinh_character_score(
        streak=streak,
        velocity=velocity,
        chop=chop,
        closing_strength_avg=closing_strength_avg,
        reversal_rate=reversal_rate,
        autocorr=autocorr,
        streak_percentile=streak_percentile,
        velocity_percentile=velocity_percentile,
    )

    nhan = gan_nhan_tinh_cach(diem, velocity_percentile, streak)
    canh_bao = kiem_tra_squat_va_churning(df_ohlcv, gia_pivot_gan_nhat)

    ket_qua = {
        "ma": ma,
        "ngay_danh_gia": _dt.date.today().isoformat(),
        "nhan_tinh_cach": nhan,
        "chi_tiet": {
            "streak_hien_tai": streak,
            "velocity_10_phien_pct": round(velocity, 3),
            "pct_change_10_phien": round(pct_change_10p, 3),
            "choppiness_index": round(chop, 1) if pd.notna(chop) else None,
            "closing_strength_trung_binh": round(closing_strength_avg, 3),
            "ty_le_dao_chieu_30_phien": round(reversal_rate, 3),
            "autocorrelation_lag1": round(autocorr, 3),
            "streak_percentile_noi_tai": round(streak_percentile, 1),
            "velocity_percentile_noi_tai": round(velocity_percentile, 1),
        },
        "character_score": diem.character_score,
        "choppiness_score": diem.choppiness_score,
        "canh_bao": canh_bao,
        "khuyen_nghi_chien_luoc": goi_y_chien_luoc_theo_nhan(nhan),
        "do_tin_cay_thap": do_tin_cay_thap,
    }
    return ket_qua


# ============================================================================
# 8 — Tiện ích tích hợp với các module khác
# ============================================================================

def he_so_chiet_khau_do_tin_cay(choppiness_score: float) -> float:
    """
    Dùng bởi stock_signal_engine.py: hệ số nhân vào độ tin cậy của tín hiệu
    Breakout khi mã đang có choppiness_score cao (dễ là breakout giả).
    """
    if choppiness_score > 1.0:
        return 0.7
    return 1.0


def gioi_han_ty_trong_theo_tinh_cach(nhan_tinh_cach: str, canh_bao: list[str], ty_trong_de_xuat: float) -> float:
    """
    Dùng bởi capital_allocator.py: giảm tỷ trọng phân bổ vốn tối đa cho phép
    nếu mã có nhãn BUNG_NO_NGAN hoặc có cờ cảnh báo CHURNING.
    """
    if nhan_tinh_cach == NHAN_BUNG_NO_NGAN or any("CHURNING" in c for c in canh_bao):
        return min(ty_trong_de_xuat, ty_trong_de_xuat * 0.5)
    return ty_trong_de_xuat
