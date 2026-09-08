"""
daily_breakout_backtest.py
============================
[Bổ sung 07/09/2026 — Chiến lược Breakout + Nén biên độ, nến NGÀY]

Chuyển thể chiến lược "Đề xuất #1" (đã backtest tốt trên NẾN H1 của hợp
đồng phái sinh VN30F1M, xem file prompt gốc) sang NẾN NGÀY cho CỔ PHIẾU.
KHÔNG áp dụng máy móc — đây là bản đã ĐÁNH GIÁ LẠI từng tham số theo đúng
yêu cầu gốc, các quyết định quan trọng:

1. **CHỈ LONG, bỏ hoàn toàn nhánh SHORT của thiết kế gốc.** Cổ phiếu
   thường (không phải phái sinh) KHÔNG có cơ chế bán khống cho nhà đầu
   tư cá nhân tại VN qua tài khoản tiền mặt thông thường — backtest
   nhánh SHORT sẽ cho kết quả không thể thực thi được trong thực tế,
   gây hiểu lầm nghiêm trọng nếu vẫn báo cáo. Nếu sau này có tài khoản
   margin hỗ trợ bán khống, có thể bổ sung lại như một chế độ riêng.

2. **Chu kỳ EMA/MA/lookback THAM SỐ HÓA** — cung cấp sẵn 2 cấu hình để
   đối chiếu qua backtest thay vì áp thẳng số gốc "150/20/2" (tính theo
   NẾN H1, không phải nến ngày):
       - `CAU_HINH_GOC`: giữ NGUYÊN số nến (EMA150/MA20/lookback=2) áp
         thẳng vào nến ngày — đơn giản nhưng "tầm nhìn" EMA150 giãn ra
         ~7 tháng thay vì ~1 tháng như bản H1 gốc.
       - `CAU_HINH_SCALE_THOI_GIAN`: scale EMA/MA theo "tầm nhìn thời
         gian" TƯƠNG ĐƯƠNG bản H1 gốc, dùng hệ số quy đổi
         `SO_NEN_H1_MOI_PHIEN` (~4 nến H1/phiên giao dịch phái sinh
         VN30F — ước tính từ giờ giao dịch liên tục 9h00-11h30 +
         13h00-14h30 ≈ 4 giờ/phiên, ĐÂY LÀ GIẢ ĐỊNH XẤP XỈ, cần điều
         chỉnh lại nếu có số liệu giờ giao dịch chính xác hơn) ->
         EMA150/4≈38, MA20/4=5. `lookback` CỐ Ý GIỮ NGUYÊN = 2 ngày ở
         CẢ 2 cấu hình — quy đổi theo giờ sẽ ra <1 ngày (không còn ý
         nghĩa "vùng nén nhiều phiên" của thiết kế gốc), nên giữ nguyên
         số ngày thay vì scale theo giờ.
   Hai cấu hình PHẢI được backtest và đối chiếu số liệu thật (xem
   `quet_watchlist_breakout` gọi 2 lần với 2 cấu hình) — không giả định
   trước cấu hình nào tốt hơn.

3. **Milestone/floor/SL/entry đều xét theo GIÁ ĐÓNG CỬA** — dữ liệu nến
   ngày không có thông tin thứ tự xảy ra trước/sau trong phiên (không
   biết High hay Low xảy ra trước), nên KHÔNG dùng High/Low nội phiên để
   suy luận đã chạm mốc chốt lời/floor hay chưa — chỉ dùng Close, nhất
   quán với cách entry/SL bản gốc đã quy định ("kiểm tra khi nến MỚI
   đóng cửa, không phải real-time"). Đây là một ĐƠN GIẢN HÓA có chủ đích
   (có thể đánh giá thấp mức lãi tối đa đã đạt được trong phiên), không
   phải giả định ngẫu nhiên.

4. **Mọi hành động thực thi (vào lệnh, SL, chốt lời từng phần, đóng do
   floor) đều Ở GIÁ MỞ CỬA PHIÊN KẾ TIẾP** sau khi phát hiện tín hiệu ở
   giá đóng cửa — tránh lookahead bias, đúng quy ước
   `backtest/backtest_engine.py` đã dùng xuyên suốt dự án.

5. **Bộ lọc THANH KHOẢN tối thiểu** trước khi tin bất kỳ tín hiệu nào —
   bài học thực tế đã nêu trong yêu cầu gốc: backtest trên dữ liệu THIẾU
   THANH KHOẢN (nến thưa giao dịch) cho kết quả sai lệch nghiêm trọng.

6. **Quản trị vốn**: bỏ hoàn toàn `margin_safety_threshold_pct` (khái
   niệm ký quỹ của phái sinh, không áp dụng cho cổ phiếu mua bằng tiền
   mặt — không vay margin trong module này). `position_size_pct_equity`
   và `daily_loss_limit_pct` là quyết định phân bổ vốn CHUNG cho cả
   danh mục, KHÔNG thuộc phạm vi module tín hiệu 1-mã này — để module
   phân bổ vốn hiện có (`core/capital_allocation_engine.py`) xử lý,
   tránh trộn 2 trách nhiệm khác nhau vào cùng 1 chỗ.

KHÔNG kết nối lệnh thật dưới bất kỳ hình thức nào — đây THUẦN TÚY là
công cụ backtest lịch sử.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.indicators import calculate_ema, calculate_ma

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


class InvalidDailyBreakoutError(ValueError):
    """Dữ liệu đầu vào không hợp lệ cho module backtest breakout nến ngày."""


def _validate_df(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise InvalidDailyBreakoutError(
            f"DataFrame thiếu các cột bắt buộc: {sorted(missing)}. "
            f"Cần đủ: {sorted(REQUIRED_COLUMNS)}."
        )


# ==============================================================================
# CẤU HÌNH THAM SỐ — 2 phương án đối chiếu (xem mục 2 ở docstring đầu file)
# ==============================================================================

SO_NEN_H1_MOI_PHIEN = 4  # GIẢ ĐỊNH XẤP XỈ — xem ghi chú docstring đầu file

CAU_HINH_GOC: dict = {
    "ema_period": 150,
    "ma_period": 20,
    "lookback": 2,
    "range_pct_max": 2.0,
    "body_pct_min": 0.3,
    "body_pct_max": 2.5,
}

CAU_HINH_SCALE_THOI_GIAN: dict = {
    "ema_period": round(150 / SO_NEN_H1_MOI_PHIEN),
    "ma_period": round(20 / SO_NEN_H1_MOI_PHIEN),
    "lookback": 2,  # CỐ Ý giữ nguyên — xem mục 2 ở docstring đầu file
    "range_pct_max": 2.0,
    "body_pct_min": 0.3,
    "body_pct_max": 2.5,
}

# (ngưỡng lãi %, % khối lượng GỐC đóng tại mốc đó) — thứ tự tăng dần,
# tổng % phải đúng bằng 100.
MOC_CHOT_LOI_TUNG_PHAN: list[tuple[float, float]] = [
    (7.5, 40.0), (8.75, 20.0), (10.0, 10.0),
    (11.25, 10.0), (13.75, 10.0), (16.25, 10.0),
]
assert sum(ty_le for _, ty_le in MOC_CHOT_LOI_TUNG_PHAN) == 100.0

# (lãi% >= mốc dưới, < mốc trên) -> khóa lãi tối thiểu (floor) — CHỈ TĂNG.
BANG_KHOA_LAI: list[tuple[float, float, float]] = [
    (0.62, 1.25, 0.25),
    (1.25, 2.5, 0.62),
    (2.5, 3.75, 1.25),
    (3.75, 6.25, 2.5),
    (6.25, float("inf"), 3.75),
]


def _tim_floor_theo_tier(lai_pct: float) -> Optional[float]:
    """Tra bảng `BANG_KHOA_LAI` — trả về mức khóa lãi (floor) ứng với mức
    lãi hiện tại, hoặc `None` nếu lãi chưa đạt 0.62% (chưa có floor)."""
    for muc_duoi, muc_tren, floor in BANG_KHOA_LAI:
        if muc_duoi <= lai_pct < muc_tren:
            return floor
    return None


# ==============================================================================
# BƯỚC 1 — Chỉ báo: EMA/MA xu hướng + biên độ nén theo lookback
# ==============================================================================

def tinh_chi_bao_breakout(
    df: pd.DataFrame, ema_period: int = 150, ma_period: int = 20, lookback: int = 2,
) -> pd.DataFrame:
    """Trả về bản sao `df` (đã sắp theo ngày tăng dần) kèm các cột:
        - `ema`, `ma`: EMA(ema_period)/MA(ma_period) trên giá đóng cửa.
        - `cao_nhat_lookback`, `thap_nhat_lookback`: mức cao nhất/thấp
          nhất của ĐÚNG `lookback` phiên LIỀN TRƯỚC (KHÔNG gồm phiên hiện
          tại) — dùng cho điều kiện breakout + nén biên độ.
    """
    _validate_df(df)
    df = df.sort_values("date").reset_index(drop=True).copy()

    df["ema"] = calculate_ema(df, ema_period)
    df["ma"] = calculate_ma(df, ma_period)
    df["cao_nhat_lookback"] = df["high"].shift(1).rolling(window=lookback, min_periods=lookback).max()
    df["thap_nhat_lookback"] = df["low"].shift(1).rolling(window=lookback, min_periods=lookback).min()
    return df


# ==============================================================================
# BƯỚC 2 — Phát hiện tín hiệu vào lệnh (CHỈ LONG — xem mục 1 đầu file)
# ==============================================================================

def phat_hien_tin_hieu_vao_lenh(
    df_co_chi_bao: pd.DataFrame,
    range_pct_max: float = 2.0,
    body_pct_min: float = 0.3,
    body_pct_max: float = 2.5,
) -> pd.DataFrame:
    """Thêm các cột tín hiệu vào `df_co_chi_bao` (đã gọi
    `tinh_chi_bao_breakout()` trước đó):
        - `bien_do_nen_pct`: biên độ dao động của `lookback` phiên trước
          = (cao_nhat_lookback - thap_nhat_lookback) / thap_nhat_lookback × 100.
        - `than_nen_pct`: "thân nến" theo ĐỊNH NGHĨA RIÊNG của chiến lược
          này = (High − Low) / Low × 100 của CHÍNH phiên hiện tại — KHÁC
          định nghĩa thông thường |Close − Open|.
        - `du_dieu_kien_vao_lenh`: đã thỏa CẢ 3 điều kiện breakout + nén
          biên độ + lọc xu hướng (CHƯA xét điều kiện thân nến).
        - `tu_dong_vao_lenh`: `du_dieu_kien_vao_lenh` VÀ thân nến nằm
          trong [body_pct_min, body_pct_max] -> đủ điều kiện tự động vào
          lệnh khi backtest.
        - `chi_canh_bao`: `du_dieu_kien_vao_lenh` VÀ thân nến VƯỢT
          body_pct_max -> chỉ mang tính cảnh báo (nghi bull-trap), KHÔNG
          tính là lệnh trong backtest.
    """
    df = df_co_chi_bao.copy()

    df["bien_do_nen_pct"] = (
        (df["cao_nhat_lookback"] - df["thap_nhat_lookback"]) / df["thap_nhat_lookback"] * 100.0
    )
    df["than_nen_pct"] = (df["high"] - df["low"]) / df["low"] * 100.0

    tin_hieu_breakout = df["close"] > df["cao_nhat_lookback"]
    dieu_kien_nen = df["bien_do_nen_pct"] <= range_pct_max
    dieu_kien_xu_huong = (df["close"] > df["ema"]) | ((df["close"] < df["ema"]) & (df["close"] > df["ma"]))

    df["du_dieu_kien_vao_lenh"] = (tin_hieu_breakout & dieu_kien_nen & dieu_kien_xu_huong).fillna(False)

    dieu_kien_than_nen_tu_dong = df["than_nen_pct"].between(body_pct_min, body_pct_max)
    dieu_kien_than_nen_canh_bao = df["than_nen_pct"] > body_pct_max

    df["tu_dong_vao_lenh"] = (df["du_dieu_kien_vao_lenh"] & dieu_kien_than_nen_tu_dong).fillna(False)
    df["chi_canh_bao"] = (df["du_dieu_kien_vao_lenh"] & dieu_kien_than_nen_canh_bao).fillna(False)
    return df


# ==============================================================================
# BƯỚC 3 — Bộ lọc thanh khoản tối thiểu
# ==============================================================================

def kiem_tra_dieu_kien_thanh_khoan(
    df: pd.DataFrame, so_phien_trung_binh: int = 20, gia_tri_trung_binh_toi_thieu: float = 1_000_000_000.0,
) -> pd.Series:
    """Trả về `pd.Series` bool — True tại các phiên có GIÁ TRỊ GIAO DỊCH
    TRUNG BÌNH (khối lượng × giá đóng cửa) trong `so_phien_trung_binh`
    phiên GẦN NHẤT (gồm cả phiên hiện tại) đạt tối thiểu
    `gia_tri_trung_binh_toi_thieu` (VND).

    LƯU Ý ĐƠN VỊ: `close` trong `ohlcv_history` của dự án đang ở đơn vị
    NGHÌN ĐỒNG (VD "25.4" = 25.400đ — xem CLAUDE.md mục 4l) — PHẢI nhân
    thêm 1000 khi quy đổi giá trị giao dịch ra VND thật, nếu không sẽ
    đánh giá SAI THẤP thanh khoản đi 1000 lần.
    """
    _validate_df(df)
    gia_tri_giao_dich = df["volume"] * df["close"] * 1000.0
    gia_tri_trung_binh = gia_tri_giao_dich.rolling(window=so_phien_trung_binh, min_periods=so_phien_trung_binh).mean()
    return (gia_tri_trung_binh >= gia_tri_trung_binh_toi_thieu).fillna(False)


# ==============================================================================
# BƯỚC 4 — Backtest: SL theo thân nến tín hiệu + chốt lời theo tầng/floor
# ==============================================================================

@dataclass
class DotDongLenh:
    """Một đợt đóng (toàn phần hoặc một phần) của 1 lệnh."""

    ngay: pd.Timestamp
    gia: float
    ty_trong_pct: float  # % của khối lượng GỐC đóng tại đợt này (0-100)
    ly_do: str  # "milestone_<mốc>" | "SL" | "floor_break" | "het_du_lieu"


@dataclass
class LenhBreakoutNgay:
    """Một lệnh LONG hoàn chỉnh (có thể đóng qua NHIỀU đợt)."""

    ngay_tin_hieu: pd.Timestamp
    ngay_vao_lenh: pd.Timestamp
    gia_vao_lenh: float
    sl: float
    than_nen_tin_hieu_pct: float
    cac_dot_dong: list[DotDongLenh] = field(default_factory=list)
    pnl_pct_tong_hop: float = 0.0  # bình quân gia quyền theo tỷ trọng đóng
    thang: bool = False


def backtest_chien_luoc_breakout(
    df: pd.DataFrame,
    cau_hinh: Optional[dict] = None,
    yeu_cau_thanh_khoan: bool = True,
    so_phien_trung_binh_thanh_khoan: int = 20,
    gia_tri_trung_binh_toi_thieu: float = 1_000_000_000.0,
) -> list[LenhBreakoutNgay]:
    """Backtest chiến lược breakout nến ngày (CHỈ LONG) trên 1 mã, trả về
    danh sách `LenhBreakoutNgay` — mỗi lệnh có thể đóng qua nhiều đợt
    (milestone chốt lời từng phần, floor, SL, hoặc hết dữ liệu).

    `cau_hinh` mặc định `CAU_HINH_GOC` nếu không truyền — LUÔN nên chạy
    backtest CẢ `CAU_HINH_GOC` và `CAU_HINH_SCALE_THOI_GIAN` rồi so sánh
    (xem mục 2 ở docstring đầu file), KHÔNG giả định trước cấu hình nào
    tốt hơn.

    Chỉ tính LÀM LỆNH THẬT các tín hiệu đủ điều kiện thân nến TỰ ĐỘNG vào
    lệnh (`tu_dong_vao_lenh`) — tín hiệu "chỉ cảnh báo" (thân nến quá to)
    KHÔNG được backtest thành lệnh, đúng tinh thần thiết kế gốc (cần xác
    nhận thủ công, không tự động).
    """
    cau_hinh = cau_hinh or CAU_HINH_GOC
    _validate_df(df)

    df2 = tinh_chi_bao_breakout(
        df, ema_period=cau_hinh["ema_period"], ma_period=cau_hinh["ma_period"],
        lookback=cau_hinh["lookback"],
    )
    df2 = phat_hien_tin_hieu_vao_lenh(
        df2, range_pct_max=cau_hinh["range_pct_max"],
        body_pct_min=cau_hinh["body_pct_min"], body_pct_max=cau_hinh["body_pct_max"],
    )

    tin_hieu_vao = df2["tu_dong_vao_lenh"].copy()
    if yeu_cau_thanh_khoan:
        thanh_khoan_ok = kiem_tra_dieu_kien_thanh_khoan(
            df2, so_phien_trung_binh=so_phien_trung_binh_thanh_khoan,
            gia_tri_trung_binh_toi_thieu=gia_tri_trung_binh_toi_thieu,
        )
        tin_hieu_vao = tin_hieu_vao & thanh_khoan_ok

    n = len(df2)
    lenh_list: list[LenhBreakoutNgay] = []
    i = 0
    while i < n - 1:  # cần tối thiểu 1 phiên kế tiếp để vào lệnh
        if not bool(tin_hieu_vao.iloc[i]):
            i += 1
            continue

        ngay_tin_hieu = df2["date"].iloc[i]
        gia_vao = float(df2["open"].iloc[i + 1])
        ngay_vao = df2["date"].iloc[i + 1]
        sl = (float(df2["high"].iloc[i]) + float(df2["low"].iloc[i])) / 2.0
        than_nen = float(df2["than_nen_pct"].iloc[i])

        remaining_pct = 100.0
        floor_val: Optional[float] = None
        milestones_hit: set[float] = set()
        cac_dot: list[DotDongLenh] = []
        ngay_dong_idx = n - 1  # mặc định: hết dữ liệu

        j = i + 1
        while j < n:
            close_j = float(df2["close"].iloc[j])
            # round(..., 9): tránh nhiễu số học dấu phẩy động (VD giá trị
            # lý thuyết đúng bằng 8.75% có thể tính ra 8.749999999999998
            # do phép chia/nhân) khiến so sánh "==biên tier" phía dưới sai
            # lệch — đã phát hiện qua test thực tế 07/09/2026.
            lai_pct = round((close_j - gia_vao) / gia_vao * 100.0, 9)

            # 1. Cắt lỗ trước tiên
            if close_j < sl:
                if j + 1 < n:
                    gia_dong, ngay_dong, idx_dong = float(df2["open"].iloc[j + 1]), df2["date"].iloc[j + 1], j + 1
                else:
                    gia_dong, ngay_dong, idx_dong = close_j, df2["date"].iloc[j], j
                cac_dot.append(DotDongLenh(ngay_dong, gia_dong, remaining_pct, "SL"))
                remaining_pct = 0.0
                ngay_dong_idx = idx_dong
                break

            # 2. Cập nhật floor (CHỈ TĂNG, không bao giờ giảm)
            tier_floor = _tim_floor_theo_tier(lai_pct)
            if tier_floor is not None:
                floor_val = tier_floor if floor_val is None else max(floor_val, tier_floor)

            # 3. Kiểm tra mốc chốt lời từng phần MỚI (chưa kích hoạt)
            for nguong, ty_le in MOC_CHOT_LOI_TUNG_PHAN:
                if nguong in milestones_hit or lai_pct < nguong:
                    continue
                milestones_hit.add(nguong)
                ty_trong_thuc_te = min(ty_le, remaining_pct)
                if ty_trong_thuc_te <= 0:
                    continue
                if j + 1 < n:
                    gia_dong, ngay_dong, idx_dong = float(df2["open"].iloc[j + 1]), df2["date"].iloc[j + 1], j + 1
                else:
                    gia_dong, ngay_dong, idx_dong = close_j, df2["date"].iloc[j], j
                cac_dot.append(DotDongLenh(ngay_dong, gia_dong, ty_trong_thuc_te, f"milestone_{nguong}"))
                remaining_pct -= ty_trong_thuc_te
                ngay_dong_idx = idx_dong
            if remaining_pct <= 1e-9:
                break

            # 4. Kiểm tra hồi về floor đã khóa
            if floor_val is not None and lai_pct <= floor_val:
                if j + 1 < n:
                    gia_dong, ngay_dong, idx_dong = float(df2["open"].iloc[j + 1]), df2["date"].iloc[j + 1], j + 1
                else:
                    gia_dong, ngay_dong, idx_dong = close_j, df2["date"].iloc[j], j
                cac_dot.append(DotDongLenh(ngay_dong, gia_dong, remaining_pct, "floor_break"))
                remaining_pct = 0.0
                ngay_dong_idx = idx_dong
                break

            j += 1

        if remaining_pct > 1e-9:
            gia_dong = float(df2["close"].iloc[-1])
            ngay_dong = df2["date"].iloc[-1]
            cac_dot.append(DotDongLenh(ngay_dong, gia_dong, remaining_pct, "het_du_lieu"))
            ngay_dong_idx = n - 1

        pnl_tong_hop = sum(
            ((dot.gia - gia_vao) / gia_vao * 100.0) * (dot.ty_trong_pct / 100.0)
            for dot in cac_dot
        )
        lenh_list.append(LenhBreakoutNgay(
            ngay_tin_hieu=ngay_tin_hieu, ngay_vao_lenh=ngay_vao, gia_vao_lenh=gia_vao,
            sl=sl, than_nen_tin_hieu_pct=than_nen, cac_dot_dong=cac_dot,
            pnl_pct_tong_hop=pnl_tong_hop, thang=pnl_tong_hop > 0,
        ))

        i = max(ngay_dong_idx, i + 1)

    return lenh_list


# ==============================================================================
# BƯỚC 5 — Tổng hợp kết quả (mô phỏng vốn TUẦN TỰ + profit factor + drawdown)
# ==============================================================================

def tong_hop_ket_qua(
    lenh_list: list[LenhBreakoutNgay], initial_capital: float = 1_000_000_000.0,
) -> dict:
    """Tổng hợp danh sách lệnh thành các chỉ số hiệu suất, CÙNG ĐỊNH DẠNG
    với báo cáo "Đề xuất #1" gốc (n_trades, win_rate, profit_factor, lãi
    cộng dồn, drawdown tối đa) để dễ đối chiếu.

    `total_return_pct`: mô phỏng vốn TUẦN TỰ (lệnh sau tính trên vốn ĐÃ
    tính lệnh trước — compounding), theo `pnl_pct_tong_hop` từng lệnh.
    `profit_factor`: tổng % lãi các lệnh THẮNG / trị tuyệt đối tổng % lỗ
    các lệnh THUA (tính trên pnl_pct_tong_hop, không phải VND tuyệt đối —
    không phụ thuộc initial_capital).
    `max_drawdown_pct`: mức sụt giảm tối đa của đường vốn tuần tự trên,
    tính theo % so với đỉnh vốn đã đạt được TRƯỚC ĐÓ.
    """
    if not lenh_list:
        return {
            "n_trades": 0, "win_rate_pct": None, "total_return_pct": None,
            "avg_return_pct": None, "profit_factor": None,
            "max_drawdown_pct": None, "ending_capital": initial_capital,
        }

    von = initial_capital
    von_theo_thoi_gian = [initial_capital]
    tong_lai_thang = 0.0
    tong_lo_thua = 0.0
    so_thang = 0

    for lenh in lenh_list:
        von *= (1 + lenh.pnl_pct_tong_hop / 100.0)
        von_theo_thoi_gian.append(von)
        if lenh.pnl_pct_tong_hop > 0:
            so_thang += 1
            tong_lai_thang += lenh.pnl_pct_tong_hop
        else:
            tong_lo_thua += lenh.pnl_pct_tong_hop

    von_series = pd.Series(von_theo_thoi_gian)
    dinh_von = von_series.cummax()
    sut_giam = (von_series - dinh_von) / dinh_von
    max_drawdown_pct = float(abs(sut_giam.min()) * 100.0)

    profit_factor = (tong_lai_thang / abs(tong_lo_thua)) if tong_lo_thua != 0 else (
        float("inf") if tong_lai_thang > 0 else None
    )

    return {
        "n_trades": len(lenh_list),
        "win_rate_pct": so_thang / len(lenh_list) * 100.0,
        "total_return_pct": (von / initial_capital - 1.0) * 100.0,
        "avg_return_pct": float(np.mean([l.pnl_pct_tong_hop for l in lenh_list])),
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
        "ending_capital": von,
    }
