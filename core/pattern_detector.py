"""
pattern_detector.py
====================
[Giai đoạn 2 — Nhận diện mô hình]

Nhận diện các mô hình kỹ thuật có đặc điểm chung: BIÊN ĐỘ DAO ĐỘNG GIÁ THU
HẸP DẦN theo thời gian (ví dụ: cốc tay cầm — cup and handle, hộp Darvas —
Darvas box).

LOGIC CHÍNH (đã chốt theo bản cập nhật mới nhất):
1. Phạm vi quét: 10-30 THÁNG gần nhất (khoảng 200-600 nến ngày).
2. Chia toàn bộ khoảng dữ liệu quét thành các đoạn liên tiếp (mặc định 4
   đoạn), tính biên độ dao động mỗi đoạn: (đỉnh - đáy) / đáy * 100%.
3. Kiểm tra biên độ có thu hẹp dần qua các đoạn không (vd: 20% -> 10% ->
   3-5%). Nếu có -> ứng viên breakout tiềm năng.
4. Thời gian hình thành càng lâu (càng gần mốc 30 tháng) -> confidence
   score càng cao.
5. Trả về: ngày bắt đầu/kết thúc từng đoạn, % biên độ từng đoạn, độ tin
   cậy tổng thể, và mức giá đỉnh vùng tích lũy (tham khảo ngưỡng breakout).

Hàm chính: detect_narrowing_pattern(df, scan_months_range=(10,30),
n_segments=4) -> dict hoặc None nếu không phát hiện mô hình phù hợp.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "high", "low"}

AVG_DAYS_PER_MONTH = 30.44  # dùng để quy đổi tháng <-> ngày lịch


class InsufficientHistoryError(ValueError):
    """Không đủ lịch sử dữ liệu (chưa đạt scan_months_min) để nhận diện mô hình."""


def _validate_df(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame thiếu các cột bắt buộc: {sorted(missing)}. "
            f"Cần đủ tối thiểu: {sorted(REQUIRED_COLUMNS)}."
        )


def _segment_amplitude(segment: pd.DataFrame) -> tuple[float, float, float]:
    """Tính (đỉnh, đáy, % biên độ) của một đoạn dữ liệu.

    % biên độ = (đỉnh - đáy) / đáy * 100.
    """
    high = float(segment["high"].max())
    low = float(segment["low"].min())
    if low <= 0:
        raise ValueError("Giá đáy phải > 0 để tính % biên độ.")
    amplitude_pct = (high - low) / low * 100.0
    return high, low, amplitude_pct


def _is_narrowing(amplitudes: list[float], tolerance_pct: float = 0.0) -> bool:
    """Kiểm tra dãy biên độ có thu hẹp dần (không tăng) qua các đoạn không.

    `tolerance_pct` cho phép sai số nhỏ (ví dụ nhiễu thị trường) — đoạn sau
    được coi là "không vi phạm xu hướng thu hẹp" nếu không lớn hơn đoạn
    trước quá `tolerance_pct`% (mặc định 0% — yêu cầu thu hẹp nghiêm ngặt).
    """
    for i in range(1, len(amplitudes)):
        allowed_max = amplitudes[i - 1] * (1 + tolerance_pct / 100.0)
        if amplitudes[i] > allowed_max:
            return False
    return True


def detect_narrowing_pattern(
    df: pd.DataFrame,
    scan_months_range: tuple[int, int] = (10, 30),
    n_segments: int = 4,
    tolerance_pct: float = 0.0,
    symbol: Optional[str] = None,
) -> Optional[dict]:
    """Quét dữ liệu giá trong phạm vi `scan_months_range` tháng gần nhất,
    chia thành `n_segments` đoạn liên tiếp, kiểm tra biên độ dao động có
    thu hẹp dần qua các đoạn hay không.

    Tham số:
        df: DataFrame OHLCV, tối thiểu cần cột 'date', 'high', 'low'
            (khuyến nghị đủ 'open', 'close', 'volume' để dùng chung với
            các module khác trong hệ thống).
        scan_months_range: (min, max) số tháng quét — mặc định (10, 30).
        n_segments: số đoạn chia — mặc định 4 (nằm trong khoảng 3-5 theo
            yêu cầu dự án).
        tolerance_pct: sai số cho phép khi kiểm tra thu hẹp (mặc định 0 —
            yêu cầu nghiêm ngặt, đoạn sau không được lớn hơn đoạn trước).
        symbol: mã cổ phiếu (tùy chọn, chỉ để đính kèm vào kết quả trả về
            cho tiện tra cứu — không ảnh hưởng logic tính toán).

    Trả về:
        None nếu không đủ dữ liệu lịch sử (chưa đạt scan_months_min) HOẶC
        biên độ không thu hẹp dần qua các đoạn (không có mẫu hình phù hợp
        — tránh báo dương tính giả).

        Nếu phát hiện mô hình, trả về dict gồm:
            - symbol
            - segments: list các dict {start_date, end_date, high, low,
              amplitude_pct}
            - confidence: điểm tin cậy tổng thể trong khoảng [0, 1]
            - accumulation_high: giá đỉnh của đoạn gần nhất (tham khảo cho
              ngưỡng breakout)
            - scan_start_date, scan_end_date
            - effective_scan_months: số tháng thực tế đã quét được
              (<= scan_months_max, dùng khi lịch sử dữ liệu ngắn hơn 30
              tháng)
    """
    _validate_df(df)
    scan_months_min, scan_months_max = scan_months_range
    if n_segments < 2:
        raise ValueError("n_segments phải >= 2 để có thể so sánh xu hướng thu hẹp.")

    df_sorted = df.sort_values("date").reset_index(drop=True)
    last_date = df_sorted["date"].iloc[-1]
    first_date = df_sorted["date"].iloc[0]

    total_days_available = (last_date - first_date).days
    total_months_available = total_days_available / AVG_DAYS_PER_MONTH

    if total_months_available < scan_months_min:
        # Chưa đủ lịch sử dữ liệu để tin cậy nhận diện mô hình.
        return None

    effective_scan_months = min(total_months_available, scan_months_max)
    scan_start_date = last_date - pd.Timedelta(
        days=effective_scan_months * AVG_DAYS_PER_MONTH
    )
    scan_start_date = max(scan_start_date, first_date)

    df_scan = df_sorted[df_sorted["date"] >= scan_start_date].reset_index(drop=True)

    if len(df_scan) < n_segments:
        # Không đủ số phiên để chia thành n_segments đoạn có ý nghĩa.
        return None

    # Chia thành n_segments đoạn liên tiếp theo thứ tự thời gian, kích
    # thước gần bằng nhau (np.array_split xử lý tốt trường hợp không chia
    # hết).
    segment_indices = np.array_split(np.arange(len(df_scan)), n_segments)

    segments: list[dict] = []
    amplitudes: list[float] = []
    for idx_group in segment_indices:
        segment_df = df_scan.iloc[idx_group]
        high, low, amplitude_pct = _segment_amplitude(segment_df)
        segments.append({
            "start_date": segment_df["date"].iloc[0],
            "end_date": segment_df["date"].iloc[-1],
            "high": high,
            "low": low,
            "amplitude_pct": round(amplitude_pct, 4),
        })
        amplitudes.append(amplitude_pct)

    if not _is_narrowing(amplitudes, tolerance_pct=tolerance_pct):
        return None

    # --- Tính confidence score ---
    # Yếu tố 1: thời gian hình thành càng lâu (càng gần scan_months_max)
    # thì độ tin cậy càng cao.
    duration_factor = effective_scan_months / scan_months_max
    duration_factor = min(max(duration_factor, 0.0), 1.0)

    # Yếu tố 2: mức độ thu hẹp — biên độ đoạn cuối càng nhỏ so với đoạn
    # đầu thì độ tin cậy càng cao.
    if amplitudes[0] > 0:
        narrowing_factor = 1.0 - (amplitudes[-1] / amplitudes[0])
    else:
        narrowing_factor = 0.0
    narrowing_factor = min(max(narrowing_factor, 0.0), 1.0)

    confidence = round(0.5 * duration_factor + 0.5 * narrowing_factor, 4)

    return {
        "symbol": symbol,
        "segments": segments,
        "confidence": confidence,
        "accumulation_high": segments[-1]["high"],
        "scan_start_date": df_scan["date"].iloc[0],
        "scan_end_date": df_scan["date"].iloc[-1],
        "effective_scan_months": round(effective_scan_months, 2),
    }
