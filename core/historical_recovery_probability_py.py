"""
historical_recovery_probability.py
=====================================
[Module 6 — mở rộng từ core/stock_character_classifier.py]

Tính XÁC SUẤT TẦN SUẤT LỊCH SỬ (empirical/historical probability) mà một mã
phục hồi sau khi rơi vào tình huống "giảm mạnh + quá bán + khối lượng đột
biến + đóng cửa yếu" — bằng cách quét TOÀN BỘ lịch sử giá để tìm tất cả các
lần trong quá khứ tình huống tương tự đã xảy ra, rồi tính tần suất thực
nghiệm (empirical frequency) xem sau đó có bao nhiêu % số lần thực sự phục
hồi.

NGUYÊN TẮC CỐT LÕI (bắt buộc tuân thủ ở MỌI nơi dùng module này):
    Đây là XÁC SUẤT TẦN SUẤT LỊCH SỬ — KHÔNG phải xác suất dự báo tương lai
    được đảm bảo. Quá khứ không chắc lặp lại, và cỡ mẫu (số lần sự kiện
    tương tự xảy ra) càng nhỏ thì độ tin cậy của con số càng thấp — PHẢI
    luôn báo cáo kèm cỡ mẫu và mức độ tin cậy thống kê tương ứng.

CHỈ ĐỌC DỮ LIỆU / TÍNH TOÁN / THỐNG KÊ — KHÔNG đặt lệnh, KHÔNG khuyến nghị
đầu tư cá nhân hóa.

Tái sử dụng tối đa hàm đã có, KHÔNG viết lại logic trùng lặp:
    - `core.stock_character_classifier.tinh_streak_hien_tai()`
    - `core.stock_character_classifier.tinh_closing_strength()`
    - `core.indicators.calculate_rsi()`
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional

import numpy as np
import pandas as pd

from core.indicators import calculate_rsi
from core.stock_character_classifier import tinh_closing_strength, tinh_streak_hien_tai

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}

# Số phiên nền tối thiểu khuyến nghị (~3 năm) để có đủ số lần lặp lại tình
# huống tương tự — dưới ngưỡng này, độ tin cậy KHÔNG được vượt quá TRUNG_BINH
# dù cỡ mẫu sự kiện tình cờ đủ lớn (mục 8.2 tài liệu gốc).
HISTORY_RECOMMENDED_MIN_SESSIONS = 750

# Số phiên đệm cần có TRƯỚC mỗi điểm quét để đủ nền tính RSI(14)/volume_MA20
# một cách đáng tin cậy (không dùng dữ liệu quá ít phiên làm nền).
WARMUP_SESSIONS = 20


class InvalidRecoveryProbabilityError(ValueError):
    """Dữ liệu đầu vào không hợp lệ cho module xác suất phục hồi lịch sử."""


# ==============================================================================
# BỘ ĐIỀU KIỆN MẶC ĐỊNH — "TÌNH HUỐNG GIẢM" CẦN QUÉT (mục 2 tài liệu gốc)
# ==============================================================================

DIEU_KIEN_MAC_DINH = {
    # Tiêu chí 1 — Tốc độ giảm giá
    "giam_toi_thieu_pct": 8.0,   # % giảm tối thiểu
    "so_phien_toi_da": 3,        # trong tối đa bao nhiêu phiên (tốc độ = giảm_pct/số_phiên)

    # Tiêu chí 2 — Quá bán
    "rsi_toi_da": 30,            # RSI(14) tại phiên mốc phải <= ngưỡng này

    # Tiêu chí 3 — Khối lượng đột biến tại phiên mốc
    "volume_ratio_toi_thieu": 1.3,  # volume phiên mốc / volume TB 20 phiên

    # Tiêu chí 5 — Bối cảnh chuỗi giảm trước đó
    "so_phien_giam_lien_tiep_toi_thieu": 3,  # streak <= -3

    # Tiêu chí 6 — Chất lượng nến tại phiên mốc (đóng cửa yếu = bán tháo)
    "closing_strength_toi_da": 0.35,
}
# Lưu ý: Tiêu chí 4 (tương quan với VNIndex/mức độ tập trung nhóm vốn hóa
# lớn) KHÔNG đưa vào bộ lọc quét vì là đặc tính CẤP THỊ TRƯỜNG, không tính
# được cho từng mã đơn lẻ — xử lý riêng ở `tinh_do_tap_trung_phuc_hoi()`.


def _validate_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise InvalidRecoveryProbabilityError("df_ohlcv rỗng hoặc None.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise InvalidRecoveryProbabilityError(f"df_ohlcv thiếu cột bắt buộc: {missing}.")


# ==============================================================================
# BƯỚC 1 — QUÉT LỊCH SỬ TÌM CÁC "PHIÊN MỐC" (Historical Pattern Scan)
# ==============================================================================

def tim_cac_diem_moc_tinh_huong_tuong_tu(df: pd.DataFrame, dieu_kien: dict) -> list[int]:
    """Quét toàn bộ `df`, trả về danh sách CÁC VỊ TRÍ (index nguyên, không
    phải nhãn ngày) mà tại đó tình huống "giảm mạnh + quá bán + volume đột
    biến + đóng cửa yếu" được thỏa mãn.

    Mỗi vị trí trả về là "phiên mốc" (anchor day) — phiên giảm sâu nhất/cuối
    cùng của chuỗi giảm — để từ đó tính phục hồi N phiên sau.

    KHÔNG LOOK-AHEAD BIAS: tại vị trí `i`, chỉ dùng dữ liệu đến phiên `i`
    (RSI, volume_MA20, streak đều tính trên `df.iloc[: i + 1]` hoặc series
    đã align đúng vị trí `i`) — không dùng bất kỳ dữ liệu nào sau đó.
    """
    _validate_df(df)

    rsi_series = calculate_rsi(df, period=14)
    volume_ma20 = df["volume"].rolling(20).mean()
    cac_vi_tri: list[int] = []

    start = dieu_kien["so_phien_toi_da"] + WARMUP_SESSIONS
    for i in range(start, len(df)):
        cua_so = df.iloc[i - dieu_kien["so_phien_toi_da"] : i + 1]
        gia_dau_cua_so = cua_so["close"].iloc[0]
        if gia_dau_cua_so == 0 or pd.isna(gia_dau_cua_so):
            continue
        pct_change = (cua_so["close"].iloc[-1] - gia_dau_cua_so) / gia_dau_cua_so * 100

        streak_tai_i = tinh_streak_hien_tai(df.iloc[: i + 1])
        rsi_tai_i = rsi_series.iloc[i]
        vol_ma_tai_i = volume_ma20.iloc[i]
        vol_ratio_tai_i = (
            df["volume"].iloc[i] / vol_ma_tai_i if vol_ma_tai_i not in (0, None) and pd.notna(vol_ma_tai_i) else np.nan
        )
        closing_strength_tai_i = tinh_closing_strength(df.iloc[i])

        dat_dieu_kien = (
            pct_change <= -dieu_kien["giam_toi_thieu_pct"]
            and streak_tai_i <= -dieu_kien["so_phien_giam_lien_tiep_toi_thieu"]
            and pd.notna(rsi_tai_i) and rsi_tai_i <= dieu_kien["rsi_toi_da"]
            and pd.notna(vol_ratio_tai_i) and vol_ratio_tai_i >= dieu_kien["volume_ratio_toi_thieu"]
            and closing_strength_tai_i <= dieu_kien["closing_strength_toi_da"]
        )
        if dat_dieu_kien:
            cac_vi_tri.append(i)

    return cac_vi_tri


def gop_cum_diem_moc(df: pd.DataFrame, cac_vi_tri: list[int], khoang_cach_toi_da: int = 3) -> list[int]:
    """Gộp các "phiên mốc" liền kề nhau thành 1 cụm — tránh đếm trùng lặp
    một đợt giảm kéo dài nhiều phiên như nhiều sự kiện riêng biệt (mục 8.4
    tài liệu gốc: "xử lý chồng lấp thời gian — overlapping windows").

    Hai vị trí được coi là CÙNG một cụm nếu khoảng cách giữa chúng <=
    `khoang_cach_toi_da` phiên. Mỗi cụm chỉ giữ lại 1 đại diện: phiên có
    `closing_strength` THẤP NHẤT trong cụm (đóng cửa yếu nhất — bán tháo
    rõ rệt nhất, đúng tinh thần "phiên giảm sâu nhất của đợt giảm").
    """
    if not cac_vi_tri:
        return []

    cac_vi_tri_sorted = sorted(cac_vi_tri)
    clusters: list[list[int]] = [[cac_vi_tri_sorted[0]]]
    for vt in cac_vi_tri_sorted[1:]:
        if vt - clusters[-1][-1] <= khoang_cach_toi_da:
            clusters[-1].append(vt)
        else:
            clusters.append([vt])

    dai_dien: list[int] = []
    for cluster in clusters:
        if len(cluster) == 1:
            dai_dien.append(cluster[0])
            continue
        best = min(cluster, key=lambda vt: tinh_closing_strength(df.iloc[vt]))
        dai_dien.append(best)

    return dai_dien


# ==============================================================================
# BƯỚC 2 — TÍNH TẦN SUẤT PHỤC HỒI THỰC NGHIỆM (Empirical Recovery Rate)
# ==============================================================================

def tinh_ty_le_phuc_hoi(
    df: pd.DataFrame, cac_vi_tri: list[int], so_phien_du_bao: int, nguong_phuc_hoi_pct: float = 0.0
) -> dict:
    """Với mỗi vị trí (phiên mốc) tìm được, tính % thay đổi giá sau
    `so_phien_du_bao` phiên kể từ đó. Đếm bao nhiêu lần vượt
    `nguong_phuc_hoi_pct` -> tính tỷ lệ % — đây là "xác suất" đúng nghĩa
    thống kê tần suất, không phải suy luận định tính.
    """
    ket_qua_tung_lan: list[float] = []
    for vi_tri in cac_vi_tri:
        if vi_tri + so_phien_du_bao >= len(df):
            continue  # bỏ qua nếu không đủ dữ liệu tương lai (mốc quá gần hiện tại)
        gia_moc = df["close"].iloc[vi_tri]
        gia_sau = df["close"].iloc[vi_tri + so_phien_du_bao]
        if gia_moc == 0 or pd.isna(gia_moc) or pd.isna(gia_sau):
            continue
        pct_thay_doi = (gia_sau - gia_moc) / gia_moc * 100
        ket_qua_tung_lan.append(pct_thay_doi)

    n = len(ket_qua_tung_lan)
    if n == 0:
        return {
            "so_lan_quan_sat": 0, "ty_le_phuc_hoi_pct": None,
            "canh_bao": "Không đủ dữ liệu lịch sử phù hợp.",
        }

    so_lan_phuc_hoi = sum(1 for x in ket_qua_tung_lan if x > nguong_phuc_hoi_pct)
    ty_le = so_lan_phuc_hoi / n * 100

    return {
        "so_lan_quan_sat": n,
        "so_lan_phuc_hoi": so_lan_phuc_hoi,
        "ty_le_phuc_hoi_pct": round(ty_le, 1),
        "pct_thay_doi_trung_binh": round(float(np.mean(ket_qua_tung_lan)), 2),
        "pct_thay_doi_trung_vi": round(float(np.median(ket_qua_tung_lan)), 2),
        "pct_thay_doi_min": round(float(min(ket_qua_tung_lan)), 2),
        "pct_thay_doi_max": round(float(max(ket_qua_tung_lan)), 2),
        "do_lech_chuan": round(float(np.std(ket_qua_tung_lan)), 2),
    }


# ==============================================================================
# HÀM CHÍNH — tinh_xac_suat_phuc_hoi_lich_su()
# ==============================================================================

def _xep_do_tin_cay(so_lan_quan_sat: int, tong_so_phien_lich_su: int) -> tuple[str, str]:
    """Xếp mức độ tin cậy thống kê theo cỡ mẫu (mục 5 tài liệu gốc), có
    tính thêm điều kiện tổng độ dài lịch sử dữ liệu đầu vào (mục 8.2):
    nếu bản thân dữ liệu lịch sử NGẮN hơn mức khuyến nghị
    (`HISTORY_RECOMMENDED_MIN_SESSIONS`), độ tin cậy KHÔNG được vượt quá
    TRUNG_BINH dù số lần sự kiện tình cờ đủ lớn.
    """
    if so_lan_quan_sat < 5:
        do_tin_cay, ghi_chu = "RAT_THAP", (
            f"Chỉ có {so_lan_quan_sat} lần quan sát trong lịch sử — cỡ mẫu quá nhỏ, "
            "không nên dùng làm căn cứ ra quyết định."
        )
    elif so_lan_quan_sat < 15:
        do_tin_cay, ghi_chu = "THAP", (
            f"Chỉ có {so_lan_quan_sat} lần quan sát — cỡ mẫu nhỏ, xem là tham khảo, "
            "không phải căn cứ chắc chắn."
        )
    elif so_lan_quan_sat < 30:
        do_tin_cay, ghi_chu = "TRUNG_BINH", (
            f"{so_lan_quan_sat} lần quan sát — cỡ mẫu ở mức chấp nhận được, vẫn nên thận trọng."
        )
    else:
        do_tin_cay, ghi_chu = "KHA_CAO", (
            f"{so_lan_quan_sat} lần quan sát — cỡ mẫu tương đối đủ để tham khảo thống kê."
        )

    if tong_so_phien_lich_su < HISTORY_RECOMMENDED_MIN_SESSIONS and do_tin_cay in ("KHA_CAO", "TRUNG_BINH"):
        do_tin_cay = "TRUNG_BINH"
        ghi_chu += (
            f" Lưu ý thêm: tổng độ dài lịch sử đầu vào chỉ có {tong_so_phien_lich_su} phiên, "
            f"ít hơn mức khuyến nghị ({HISTORY_RECOMMENDED_MIN_SESSIONS} phiên/~3 năm) — "
            "đã giới hạn độ tin cậy tối đa ở mức TRUNG_BÌNH."
        )

    return do_tin_cay, ghi_chu


def tinh_xac_suat_phuc_hoi_lich_su(
    ma: str,
    df_ohlcv: pd.DataFrame,
    dieu_kien_loc: Optional[dict] = None,
    cac_so_phien_du_bao: tuple[int, ...] = (1, 3, 5),
    nguong_phuc_hoi_pct: float = 0.0,
    gop_cum: bool = True,
) -> dict:
    """Hàm chính: tính xác suất phục hồi lịch sử (empirical) cho 1 mã, dựa
    trên TẦN SUẤT các lần trong quá khứ tình huống tương tự đã dẫn đến
    phục hồi.

    KHÔNG phải dự báo — là thống kê tần suất quá khứ, luôn báo cáo kèm cỡ
    mẫu (`so_lan_quan_sat_lich_su`) để người dùng tự đánh giá độ tin cậy.
    """
    _validate_df(df_ohlcv)

    dieu_kien = {**DIEU_KIEN_MAC_DINH, **(dieu_kien_loc or {})}

    cac_vi_tri = tim_cac_diem_moc_tinh_huong_tuong_tu(df_ohlcv, dieu_kien)
    if gop_cum:
        cac_vi_tri = gop_cum_diem_moc(df_ohlcv, cac_vi_tri)

    ket_qua_theo_so_phien = {}
    for n_phien in cac_so_phien_du_bao:
        ket_qua_theo_so_phien[f"sau_{n_phien}_phien"] = tinh_ty_le_phuc_hoi(
            df_ohlcv, cac_vi_tri, n_phien, nguong_phuc_hoi_pct
        )

    so_lan_quan_sat = len(cac_vi_tri)
    do_tin_cay, ghi_chu_do_tin_cay = _xep_do_tin_cay(so_lan_quan_sat, len(df_ohlcv))

    return {
        "ma": ma,
        "dieu_kien_da_dung": dieu_kien,
        "so_lan_quan_sat_lich_su": so_lan_quan_sat,
        "tong_so_phien_du_lieu_dau_vao": len(df_ohlcv),
        "do_tin_cay_thong_ke": do_tin_cay,
        "ghi_chu_do_tin_cay": ghi_chu_do_tin_cay,
        "ket_qua_theo_so_phien_du_bao": ket_qua_theo_so_phien,
        "ngay_danh_gia": _dt.date.today().isoformat(),
        "canh_bao_phap_ly": (
            "Đây là TẦN SUẤT THỰC NGHIỆM từ dữ liệu lịch sử, KHÔNG phải xác suất "
            "dự báo được đảm bảo cho tương lai. Không dùng làm căn cứ duy nhất để "
            "ra quyết định giao dịch."
        ),
    }


# ==============================================================================
# MỞ RỘNG CẤP THỊ TRƯỜNG — Tiêu chí 4 (tương quan với VNIndex)
# ==============================================================================

def tinh_do_tap_trung_phuc_hoi(
    diem_dong_gop_tung_ma: dict[str, float],
) -> dict:
    """Tính % đóng góp của 3 mã dẫn đầu vào tổng điểm tăng của VNIndex
    phiên đó. Tỷ lệ càng cao (ví dụ: 2 mã lớn chiếm ~67% mức tăng) ->
    phục hồi càng "tập trung", ít lan tỏa -> độ bền của nhịp hồi thường
    thấp hơn.

    `diem_dong_gop_tung_ma`: dict {"VIC": 9.08, "VHM": 6.98, ...} — điểm
    đóng góp của từng mã vào VNIndex tại phiên phục hồi đang xét.
    """
    if not diem_dong_gop_tung_ma:
        raise InvalidRecoveryProbabilityError("diem_dong_gop_tung_ma rỗng hoặc None.")

    tong_diem_tang = sum(v for v in diem_dong_gop_tung_ma.values() if v > 0)
    top_3 = sorted(diem_dong_gop_tung_ma.values(), reverse=True)[:3]
    ty_le_top3 = sum(top_3) / tong_diem_tang * 100 if tong_diem_tang else 0.0

    return {
        "ty_le_dong_gop_top3_pct": round(ty_le_top3, 1),
        "nhan_dinh": (
            "TAP_TRUNG_CAO — thận trọng, độ bền nhịp hồi thấp"
            if ty_le_top3 > 50 else
            "LAN_TOA_TOT — độ tin cậy nhịp hồi cao hơn"
        ),
    }
