"""
experimental/indicator_lab.py
================================
"Phòng thí nghiệm chỉ báo" — module NGHIÊN CỨU/THỬ NGHIỆM, TÁCH BIỆT
HOÀN TOÀN khỏi hệ thống tín hiệu/backtest THẬT của dự án (KHÔNG sửa và
KHÔNG được import bởi bất kỳ file nào trong `core/`).

Dùng để thử nghiệm bộ tham số breakout + 4 bộ lọc chỉ báo tùy chọn (RSI/
Volume/ATR/Bollinger) trên nến NGÀY, tìm bộ tham số cho lợi nhuận ròng
cao nhất — kèm công cụ quét toàn bộ watchlist tìm mã đang thỏa tín hiệu.

⚠️ CẢNH BÁO NGHIÊN CỨU (nhắc lại ở mọi nơi hiển thị kết quả module này):
Đây là công cụ dò nhiều tổ hợp tham số trên CÙNG 1 bộ dữ liệu lịch sử —
có rủi ro "khớp quá vừa" (overfit). Hiệu suất tốt trong quá khứ KHÔNG
đảm bảo lặp lại trong tương lai. Kết quả CHƯA tính phí giao dịch/trượt
giá. Kết quả chạy thử KHÔNG được lưu vào bất kỳ nơi lưu trữ lâu dài nào
— chỉ tồn tại tạm thời trong phiên làm việc.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.indicators import calculate_bollinger_bands, calculate_ema, calculate_ma, calculate_rsi, calculate_volume_ma
from core.market_breadth import calculate_atr


class InvalidIndicatorLabError(ValueError):
    """Dữ liệu/tham số đầu vào không hợp lệ cho Phòng thí nghiệm chỉ báo."""


# ==============================================================================
# THAM SỐ MẶC ĐỊNH
# ==============================================================================

THAM_SO_MAC_DINH = {
    "buy_lookback": 4,
    "sell_lookback": 4,
    "ema_period": 200,
    "ma_period": 20,
    "range_pct_max": 5.0,
    "body_pct_min": 1.0,
    "body_pct_max": 6.0,
}

BO_LOC_MAC_DINH = {
    "rsi_enabled": False, "rsi_period": 14,
    "volume_enabled": False, "volume_period": 20,
    "atr_enabled": False, "atr_period": 14,
    "bollinger_enabled": False, "bollinger_period": 20, "bollinger_num_std": 2.0,
}

TRAILING_TP_TIERS_MAC_DINH = [
    {"muc_lai_pct": 5.0, "chot_pct_khoi_luong": 30.0},
    {"muc_lai_pct": 10.0, "chot_pct_khoi_luong": 30.0},
    {"muc_lai_pct": 15.0, "chot_pct_khoi_luong": 40.0},
]

# Số phiên "khóa" tối thiểu sau khi mua trước khi được phép bán ra —
# mô phỏng quy tắc thanh toán T+2 của TTCK Việt Nam (cổ phiếu mua về cần
# đủ 2 phiên mới thực sự về tài khoản, giao dịch/bán lại được). Chỉnh
# lại nếu quy định T+ thay đổi trong tương lai.
SO_PHIEN_KHOA_TOI_THIEU_MAC_DINH = 2

# --- Bổ sung 06/08/2026 — 3 đặc thù TTCK Việt Nam khác ---
PHI_MOI_GIOI_PCT_MAC_DINH = 0.15   # % mỗi CHIỀU (mua VÀ bán), tham khảo mức phổ biến 0,15-0,35%
THUE_BAN_PCT_MAC_DINH = 0.1        # % TRÊN GIÁ TRỊ BÁN (thuế TNCN chuyển nhượng chứng khoán, chỉ áp cho chiều BÁN)
BIEN_DO_DAO_DONG_PCT_MAC_DINH = 7.0  # % — mặc định HOSE (HNX ~10%, UPCOM ~15%, chỉnh nếu cần)
LO_GIAO_DICH = 100                 # số cổ phiếu tối thiểu/lô — số lượng luôn làm tròn XUỐNG theo lô này

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


def _lam_tron_lo(so_luong: float, lo: int = LO_GIAO_DICH) -> int:
    """Làm tròn XUỐNG theo lô giao dịch (mặc định 100 cổ phiếu) — số
    lượng lẻ dưới 1 lô coi như KHÔNG giao dịch được phần đó."""
    return int(so_luong // lo) * lo


def _kiem_tra_gan_bien_do(gia: float, gia_tham_chieu: Optional[float], bien_do_pct: float, dung_sai_pct: float = 0.5) -> Optional[str]:
    """Kiểm tra `gia` có đang NẰM SÁT biên độ dao động (trần/sàn) so với
    `gia_tham_chieu` (thường là giá đóng cửa phiên liền trước) hay không
    — trả về "gan_tran"/"gan_san"/None. CHỈ mang tính CẢNH BÁO (khả năng
    khớp lệnh thực tế có thể khó khăn do dư mua/dư bán tại trần/sàn) —
    KHÔNG thay đổi kết quả tính toán P&L.
    """
    if gia_tham_chieu is None or gia_tham_chieu <= 0:
        return None
    do_lech_pct = (gia - gia_tham_chieu) / gia_tham_chieu * 100
    if do_lech_pct >= bien_do_pct - dung_sai_pct:
        return "gan_tran"
    if do_lech_pct <= -(bien_do_pct - dung_sai_pct):
        return "gan_san"
    return None


def _validate_df(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise InvalidIndicatorLabError("df_ohlcv rỗng hoặc None.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise InvalidIndicatorLabError(f"df_ohlcv thiếu cột bắt buộc: {missing}.")


def _validate_tp_tiers(tp_tiers: list[dict]) -> None:
    if not tp_tiers:
        raise InvalidIndicatorLabError("tp_tiers không được rỗng.")
    tong_pct = sum(t["chot_pct_khoi_luong"] for t in tp_tiers)
    if tong_pct > 100.0 + 1e-6:
        raise InvalidIndicatorLabError(
            f"Tổng % khối lượng chốt qua các tier ({tong_pct}%) không được vượt quá 100%."
        )
    muc_lai = [t["muc_lai_pct"] for t in tp_tiers]
    if muc_lai != sorted(muc_lai):
        raise InvalidIndicatorLabError("Các tier trailing TP phải sắp theo mức lãi % TĂNG DẦN.")


# ==============================================================================
# 1. CÔNG THỨC NẾN CƠ BẢN
# ==============================================================================

def candle_body_pct(df: pd.DataFrame, i: int) -> Optional[float]:
    """"Thân nến" = (High - Low) / Low * 100 — KHÔNG phải |Close - Open|."""
    low = df["low"].iloc[i]
    high = df["high"].iloc[i]
    if pd.isna(low) or pd.isna(high) or low == 0:
        return None
    return float((high - low) / low * 100)


def candle_range_pct(df: pd.DataFrame, i: int, n: int) -> Optional[float]:
    """Biên độ dao động của `n` nến LIỀN TRƯỚC nến `i` (không gồm i)."""
    if i - n < 0:
        return None
    window = df.iloc[i - n: i]
    if window.empty:
        return None
    lo = window["low"].min()
    hi = window["high"].max()
    if pd.isna(lo) or pd.isna(hi) or lo == 0:
        return None
    return float((hi - lo) / lo * 100)


# ==============================================================================
# 2. TÍNH TOÁN CHỈ BÁO (1 LẦN TRÊN TOÀN CHUỖI) — TÁI SỬ DỤNG core/indicators.py
#    và core/market_breadth.py, KHÔNG viết lại logic đã có.
# ==============================================================================

def tinh_toan_chi_bao(df: pd.DataFrame, tham_so: dict, bo_loc: dict) -> dict:
    _validate_df(df)
    ket_qua = {
        "ema": calculate_ema(df, tham_so["ema_period"]),
        "ma": calculate_ma(df, tham_so["ma_period"]),
    }
    if bo_loc.get("rsi_enabled"):
        ket_qua["rsi"] = calculate_rsi(df, bo_loc.get("rsi_period", 14))
    if bo_loc.get("volume_enabled"):
        ket_qua["volume_ma"] = calculate_volume_ma(df, bo_loc.get("volume_period", 20))
    if bo_loc.get("atr_enabled"):
        atr_series = calculate_atr(df, bo_loc.get("atr_period", 14))
        ket_qua["atr"] = atr_series
        gia_tri_hop_le = atr_series.dropna().values
        ket_qua["atr_median"] = float(np.median(gia_tri_hop_le)) if len(gia_tri_hop_le) else None
    if bo_loc.get("bollinger_enabled"):
        upper, middle, lower = calculate_bollinger_bands(
            df, bo_loc.get("bollinger_period", 20), bo_loc.get("bollinger_num_std", 2.0)
        )
        ket_qua["bb_upper"] = upper
        ket_qua["bb_lower"] = lower
    return ket_qua


def _so_phien_khoi_dong_toi_thieu(tham_so: dict, bo_loc: dict) -> int:
    """Số phiên nền tối thiểu cần có TRƯỚC khi bắt đầu xét tín hiệu — đủ
    cho lookback + toàn bộ chỉ báo (EMA/MA + các bộ lọc đang bật)."""
    can = [tham_so["buy_lookback"], tham_so["sell_lookback"], tham_so["ema_period"], tham_so["ma_period"]]
    if bo_loc.get("rsi_enabled"):
        can.append(bo_loc.get("rsi_period", 14))
    if bo_loc.get("volume_enabled"):
        can.append(bo_loc.get("volume_period", 20))
    if bo_loc.get("atr_enabled"):
        can.append(bo_loc.get("atr_period", 14))
    if bo_loc.get("bollinger_enabled"):
        can.append(bo_loc.get("bollinger_period", 20))
    return max(can)


# ==============================================================================
# 3. ĐIỀU KIỆN TÍN HIỆU GỐC (mục 1 prompt) + 4 BỘ LỌC BỔ SUNG (mục 3 prompt)
# ==============================================================================

def kiem_tra_dieu_kien_buy_goc(df: pd.DataFrame, i: int, tham_so: dict, chi_bao: dict) -> Optional[bool]:
    """None = chưa đủ dữ liệu để đánh giá. True/False = kết quả rõ ràng."""
    n = tham_so["buy_lookback"]
    if i - n < 0:
        return None
    window = df.iloc[i - n: i]
    high_max = window["high"].max()
    close_i = df["close"].iloc[i]
    if pd.isna(high_max) or pd.isna(close_i):
        return None
    dk1_breakout = close_i > high_max

    range_pct = candle_range_pct(df, i, n)
    if range_pct is None:
        return None
    dk2_nen = range_pct <= tham_so["range_pct_max"]

    ema_i, ma_i = chi_bao["ema"].iloc[i], chi_bao["ma"].iloc[i]
    if pd.isna(ema_i) or pd.isna(ma_i):
        return None
    dk3_xu_huong = (close_i > ema_i) or (close_i < ema_i and close_i > ma_i)

    if not (dk1_breakout and dk2_nen and dk3_xu_huong):
        return False

    body_pct = candle_body_pct(df, i)
    if body_pct is None:
        return False
    return tham_so["body_pct_min"] <= body_pct <= tham_so["body_pct_max"]


def kiem_tra_dieu_kien_sell_goc(df: pd.DataFrame, i: int, tham_so: dict, chi_bao: dict) -> Optional[bool]:
    n = tham_so["sell_lookback"]
    if i - n < 0:
        return None
    window = df.iloc[i - n: i]
    low_min = window["low"].min()
    close_i = df["close"].iloc[i]
    if pd.isna(low_min) or pd.isna(close_i):
        return None
    dk1_breakdown = close_i < low_min

    range_pct = candle_range_pct(df, i, n)
    if range_pct is None:
        return None
    dk2_nen = range_pct <= tham_so["range_pct_max"]

    ema_i, ma_i = chi_bao["ema"].iloc[i], chi_bao["ma"].iloc[i]
    if pd.isna(ema_i) or pd.isna(ma_i):
        return None
    dk3_xu_huong = (close_i < ema_i) or (close_i > ema_i and close_i < ma_i)

    if not (dk1_breakdown and dk2_nen and dk3_xu_huong):
        return False

    body_pct = candle_body_pct(df, i)
    if body_pct is None:
        return False
    return tham_so["body_pct_min"] <= body_pct <= tham_so["body_pct_max"]


def kiem_tra_bo_loc_bo_sung(df: pd.DataFrame, i: int, chi_bao: dict, bo_loc: dict, huong: str) -> bool:
    """4 bộ lọc thêm (RSI/Volume/ATR/Bollinger) — kết hợp VÀ (AND) với
    điều kiện gốc. `huong` = "BUY" hoặc "SELL"."""
    if bo_loc.get("rsi_enabled"):
        rsi_i = chi_bao["rsi"].iloc[i]
        if pd.isna(rsi_i):
            return False
        if huong == "BUY" and not (rsi_i > 50):
            return False
        if huong == "SELL" and not (rsi_i < 50):
            return False

    if bo_loc.get("volume_enabled"):
        vol_i = df["volume"].iloc[i]
        vol_ma_i = chi_bao["volume_ma"].iloc[i]
        if pd.isna(vol_i) or pd.isna(vol_ma_i) or not (vol_i > vol_ma_i):
            return False

    if bo_loc.get("atr_enabled"):
        atr_i = chi_bao["atr"].iloc[i]
        atr_median = chi_bao.get("atr_median")
        if pd.isna(atr_i) or atr_median is None or not (atr_i > atr_median):
            return False

    if bo_loc.get("bollinger_enabled"):
        close_i = df["close"].iloc[i]
        if huong == "BUY":
            bb_upper_i = chi_bao["bb_upper"].iloc[i]
            if pd.isna(bb_upper_i) or not (close_i > bb_upper_i):
                return False
        else:
            bb_lower_i = chi_bao["bb_lower"].iloc[i]
            if pd.isna(bb_lower_i) or not (close_i < bb_lower_i):
                return False

    return True


def danh_gia_tin_hieu(df: pd.DataFrame, i: int, tham_so: dict, bo_loc: dict, chi_bao: dict) -> Optional[str]:
    """Trả về "BUY", "SELL", hoặc None (không có tín hiệu / là "trap")."""
    ket_qua_buy = kiem_tra_dieu_kien_buy_goc(df, i, tham_so, chi_bao)
    if ket_qua_buy and kiem_tra_bo_loc_bo_sung(df, i, chi_bao, bo_loc, "BUY"):
        return "BUY"

    ket_qua_sell = kiem_tra_dieu_kien_sell_goc(df, i, tham_so, chi_bao)
    if ket_qua_sell and kiem_tra_bo_loc_bo_sung(df, i, chi_bao, bo_loc, "SELL"):
        return "SELL"

    return None


def danh_gia_tin_hieu_ket_hop(
    df: pd.DataFrame, i: int, danh_sach_tham_so: list[dict], bo_loc: dict, danh_sach_chi_bao: list[dict],
) -> Optional[str]:
    """Đánh giá tín hiệu KẾT HỢP nhiều bộ tham số theo logic OR (bổ sung
    06/08/2026) — trả về "BUY" nếu ÍT NHẤT 1 trong các bộ tham số thỏa
    điều kiện BUY tại nến `i` (mỗi bộ dùng ĐÚNG chỉ báo EMA/MA riêng của
    nó, vì các bộ có thể dùng chu kỳ khác nhau).

    Dùng để mô phỏng chiến lược "vào lệnh nếu mã đạt tiêu chí của BẤT KỲ
    bộ nào trong N bộ tham số đã chọn" — tăng số lượng tín hiệu (hợp của
    nhiều bộ lọc) so với chỉ dùng ĐÚNG 1 bộ duy nhất.
    """
    for tham_so, chi_bao in zip(danh_sach_tham_so, danh_sach_chi_bao):
        ket_qua_buy = kiem_tra_dieu_kien_buy_goc(df, i, tham_so, chi_bao)
        if ket_qua_buy and kiem_tra_bo_loc_bo_sung(df, i, chi_bao, bo_loc, "BUY"):
            return "BUY"

    for tham_so, chi_bao in zip(danh_sach_tham_so, danh_sach_chi_bao):
        ket_qua_sell = kiem_tra_dieu_kien_sell_goc(df, i, tham_so, chi_bao)
        if ket_qua_sell and kiem_tra_bo_loc_bo_sung(df, i, chi_bao, bo_loc, "SELL"):
            return "SELL"

    return None


# ==============================================================================
# 4. ENGINE CHẠY THỬ (BACKTEST) — quản lý vị thế, SL, trailing TP bậc thang
# ==============================================================================

def _mo_lenh_moi(
    side: str, i: int, df: pd.DataFrame, so_tiers: int,
    von_tham_chieu: float, ty_trong_von_pct: float,
    bien_do_dao_dong_pct: float,
) -> dict:
    entry_price = float(df["close"].iloc[i])
    gia_tham_chieu_hom_truoc = float(df["close"].iloc[i - 1]) if i > 0 else None

    # Số cổ phiếu ƯỚC TÍNH (làm tròn XUỐNG lô 100) — dùng `von_tham_chieu`
    # (vốn ban đầu cố định, KHÔNG phải equity đang lãi/lỗ dồn tích) làm
    # mốc minh họa nhất quán cho MỌI lệnh, để biết thực tế mua được BAO
    # NHIÊU CỔ PHIẾU với tỷ trọng vốn đã chọn — chỉ mang tính THAM KHẢO,
    # KHÔNG ảnh hưởng tới cách tính "lợi nhuận ròng" (vẫn compound theo
    # % như trước, xem `tinh_loi_nhuan_rong`).
    von_du_kien = von_tham_chieu * ty_trong_von_pct / 100
    so_co_phieu_uoc_tinh = _lam_tron_lo(von_du_kien / entry_price) if entry_price > 0 else 0

    return {
        "side": side,
        "entry_idx": i,
        "entry_date": str(df["date"].iloc[i]),
        "entry_price": entry_price,
        "body_mid": float((df["high"].iloc[i] + df["low"].iloc[i]) / 2),
        "remaining_pct": 100.0,
        "tiers_kich_hoat": [False] * so_tiers,
        "pnl_tich_luy_co_trong_so": 0.0,
        "so_co_phieu_uoc_tinh": so_co_phieu_uoc_tinh,
        "canh_bao_bien_do_vao_lenh": _kiem_tra_gan_bien_do(entry_price, gia_tham_chieu_hom_truoc, bien_do_dao_dong_pct),
    }


def _tinh_pnl_hien_tai_pct(vi_the: dict, gia_hien_tai: float) -> float:
    if vi_the["side"] == "LONG":
        return (gia_hien_tai - vi_the["entry_price"]) / vi_the["entry_price"] * 100
    return (vi_the["entry_price"] - gia_hien_tai) / vi_the["entry_price"] * 100


def _hoan_tat_dong_lenh(
    vi_the: dict, i: int, df: pd.DataFrame, pnl_pct_cho_phan_con_lai: float,
    chi_phi_giao_dich_pct: float, bien_do_dao_dong_pct: float,
) -> dict:
    """Đóng lệnh (do chạm Stop Loss). `chi_phi_giao_dich_pct` = tổng phí
    môi giới (2 chiều) + thuế bán — đã CHỨNG MINH tương đương chính xác
    với việc trừ TỪNG PHẦN theo trọng số ở mỗi tier (vì tổng trọng số
    luôn = 100%), nên chỉ cần trừ 1 LẦN vào kết quả cuối cùng."""
    vi_the["pnl_tich_luy_co_trong_so"] += vi_the["remaining_pct"] * pnl_pct_cho_phan_con_lai
    final_pnl_pct_truoc_phi = vi_the["pnl_tich_luy_co_trong_so"] / 100
    final_pnl_pct = round(final_pnl_pct_truoc_phi - chi_phi_giao_dich_pct, 2)

    gia_tham_chieu_hom_truoc = float(df["close"].iloc[i - 1]) if i > 0 else None
    exit_price_thi_truong = float(df["close"].iloc[i])  # giá THỊ TRƯỜNG thật (chưa trừ phí) — dùng để cảnh báo biên độ
    if vi_the["side"] == "LONG":
        exit_price = vi_the["entry_price"] * (1 + final_pnl_pct / 100)
    else:
        exit_price = vi_the["entry_price"] * (1 - final_pnl_pct / 100)
    return {
        "side": vi_the["side"],
        "entry_date": vi_the["entry_date"],
        "entry_price": round(vi_the["entry_price"], 2),
        "exit_date": str(df["date"].iloc[i]),
        "exit_price": round(exit_price, 2),
        "final_pnl_pct": final_pnl_pct,
        "so_co_phieu_uoc_tinh": vi_the["so_co_phieu_uoc_tinh"],
        "canh_bao_bien_do_vao_lenh": vi_the["canh_bao_bien_do_vao_lenh"],
        "canh_bao_bien_do_ra_lenh": _kiem_tra_gan_bien_do(exit_price_thi_truong, gia_tham_chieu_hom_truoc, bien_do_dao_dong_pct),
    }


def chay_backtest(
    df_ohlcv: pd.DataFrame,
    tham_so: dict,
    bo_loc: dict,
    tp_tiers: list[dict],
    von_ban_dau: float = 1_000_000_000,
    ty_trong_von_pct: float = 50.0,
    so_phien_khoa_toi_thieu: int = SO_PHIEN_KHOA_TOI_THIEU_MAC_DINH,
    phi_moi_gioi_pct: float = PHI_MOI_GIOI_PCT_MAC_DINH,
    thue_ban_pct: float = THUE_BAN_PCT_MAC_DINH,
    bien_do_dao_dong_pct: float = BIEN_DO_DAO_DONG_PCT_MAC_DINH,
) -> dict:
    """Chạy backtest tuần tự trên TOÀN BỘ `df_ohlcv` theo đúng bộ tham số/
    bộ lọc/trailing TP truyền vào. KHÔNG ghi vào storage — chỉ trả về dict
    kết quả để lớp gọi (dashboard) tự lưu tạm vào `st.session_state`.

    ĐIỀU CHỈNH CHO ĐÚNG THỰC TẾ TTCK VIỆT NAM (bổ sung 06/08/2026):
      1) CHỈ giao dịch LONG (mua) — cổ phiếu thường KHÔNG có cơ chế bán
         khống (short-sell). Tín hiệu SELL KHÔNG còn dùng để MỞ vị thế
         mới — chỉ còn là CẢNH BÁO kỹ thuật khi đang giữ LONG.
      2) `so_phien_khoa_toi_thieu` (mặc định 2, mô phỏng T+2): trong
         khoảng này kể từ ngày vào lệnh, vị thế bị KHÓA — không kiểm
         tra SL/TP dù giá đã chạm ngưỡng nào.
      3) `phi_moi_gioi_pct` (mặc định 0,15%/chiều) + `thue_ban_pct`
         (mặc định 0,1%, chỉ áp chiều bán) — trừ vào kết quả mỗi lệnh
         ĐÃ ĐÓNG (tổng phí mua + phí bán + thuế bán, trừ 1 LẦN — đã
         chứng minh tương đương chính xác với trừ theo trọng số từng
         tier vì tổng trọng số luôn = 100%).
      4) `bien_do_dao_dong_pct` (mặc định 7%, HOSE) — CHỈ dùng để CẢNH
         BÁO khi giá vào/ra lệnh nằm sát trần/sàn (khả năng khớp lệnh
         thực tế có thể khó khăn), KHÔNG thay đổi kết quả P&L.
      5) `so_co_phieu_uoc_tinh` (mỗi lệnh) — số cổ phiếu MUA ĐƯỢC (làm
         tròn XUỐNG lô 100), tính trên `von_ban_dau` cố định làm mốc —
         CHỈ mang tính THAM KHẢO, KHÔNG ảnh hưởng cách tính lợi nhuận
         ròng (vẫn compound theo %, xem `tinh_loi_nhuan_rong`).
    """
    _validate_df(df_ohlcv)
    _validate_tp_tiers(tp_tiers)
    if von_ban_dau <= 0:
        raise InvalidIndicatorLabError("von_ban_dau phải > 0.")
    if not (0 < ty_trong_von_pct <= 100):
        raise InvalidIndicatorLabError("ty_trong_von_pct phải trong khoảng (0, 100].")
    if so_phien_khoa_toi_thieu < 0:
        raise InvalidIndicatorLabError("so_phien_khoa_toi_thieu phải >= 0.")
    if phi_moi_gioi_pct < 0 or thue_ban_pct < 0:
        raise InvalidIndicatorLabError("phi_moi_gioi_pct và thue_ban_pct phải >= 0.")
    if bien_do_dao_dong_pct <= 0:
        raise InvalidIndicatorLabError("bien_do_dao_dong_pct phải > 0.")

    chi_phi_giao_dich_pct = phi_moi_gioi_pct * 2 + thue_ban_pct  # phí mua + phí bán + thuế bán

    df = df_ohlcv.reset_index(drop=True)
    chi_bao = tinh_toan_chi_bao(df, tham_so, bo_loc)
    n_rows = len(df)
    start_idx = _so_phien_khoi_dong_toi_thieu(tham_so, bo_loc)

    trades: list[dict] = []
    canh_bao_tin_hieu_nguoc: list[dict] = []
    vi_the: Optional[dict] = None

    for i in range(start_idx, n_rows):
        close_i = float(df["close"].iloc[i])

        if vi_the is not None:
            da_du_khoa_T = (i - vi_the["entry_idx"]) >= so_phien_khoa_toi_thieu

            if not da_du_khoa_T:
                pass
            else:
                # --- 1) Kiểm tra Stop Loss trước ---
                if close_i < vi_the["body_mid"]:
                    pnl_pct = _tinh_pnl_hien_tai_pct(vi_the, close_i)
                    trades.append(_hoan_tat_dong_lenh(
                        vi_the, i, df, pnl_pct, chi_phi_giao_dich_pct, bien_do_dao_dong_pct,
                    ))
                    vi_the = None
                else:
                    # --- 2) Trailing Take-Profit theo bậc thang ---
                    pnl_hien_tai = _tinh_pnl_hien_tai_pct(vi_the, close_i)
                    for idx_tier, tier in enumerate(tp_tiers):
                        if vi_the["tiers_kich_hoat"][idx_tier]:
                            continue
                        if pnl_hien_tai >= tier["muc_lai_pct"]:
                            vi_the["tiers_kich_hoat"][idx_tier] = True
                            pct_chot = tier["chot_pct_khoi_luong"]
                            vi_the["pnl_tich_luy_co_trong_so"] += pct_chot * pnl_hien_tai
                            vi_the["remaining_pct"] -= pct_chot
                            if vi_the["remaining_pct"] <= 1e-9:
                                final_pnl_pct_truoc_phi = vi_the["pnl_tich_luy_co_trong_so"] / 100
                                final_pnl_pct = round(final_pnl_pct_truoc_phi - chi_phi_giao_dich_pct, 2)
                                exit_price = vi_the["entry_price"] * (1 + final_pnl_pct / 100)
                                gia_tham_chieu_hom_truoc = float(df["close"].iloc[i - 1]) if i > 0 else None
                                trades.append({
                                    "side": "LONG", "entry_date": vi_the["entry_date"],
                                    "entry_price": round(vi_the["entry_price"], 2),
                                    "exit_date": str(df["date"].iloc[i]), "exit_price": round(exit_price, 2),
                                    "final_pnl_pct": final_pnl_pct,
                                    "so_co_phieu_uoc_tinh": vi_the["so_co_phieu_uoc_tinh"],
                                    "canh_bao_bien_do_vao_lenh": vi_the["canh_bao_bien_do_vao_lenh"],
                                    "canh_bao_bien_do_ra_lenh": _kiem_tra_gan_bien_do(close_i, gia_tham_chieu_hom_truoc, bien_do_dao_dong_pct),
                                })
                                vi_the = None
                                break

        if vi_the is None:
            tin_hieu = danh_gia_tin_hieu(df, i, tham_so, bo_loc, chi_bao)
            if tin_hieu == "BUY":
                vi_the = _mo_lenh_moi(
                    "LONG", i, df, len(tp_tiers), von_ban_dau, ty_trong_von_pct, bien_do_dao_dong_pct,
                )
            # tin_hieu == "SELL" khi KHÔNG đang giữ lệnh -> KHÔNG làm gì
            # (không mở SHORT, vì cổ phiếu thường không bán khống được).
        else:
            tin_hieu = danh_gia_tin_hieu(df, i, tham_so, bo_loc, chi_bao)
            if tin_hieu == "SELL":
                canh_bao_tin_hieu_nguoc.append({
                    "ngay": str(df["date"].iloc[i]), "tin_hieu_nguoc": tin_hieu,
                    "dang_giu_lenh": vi_the["side"],
                })

    open_position = None
    if vi_the is not None:
        close_cuoi = float(df["close"].iloc[-1])
        open_position = {
            "side": vi_the["side"],
            "entry_date": vi_the["entry_date"],
            "entry_price": round(vi_the["entry_price"], 2),
            "as_of_date": str(df["date"].iloc[-1]),
            "current_price": round(close_cuoi, 2),
            "unrealized_pnl_pct": round(_tinh_pnl_hien_tai_pct(vi_the, close_cuoi), 2),
            "so_co_phieu_uoc_tinh": vi_the["so_co_phieu_uoc_tinh"],
            "canh_bao_bien_do_vao_lenh": vi_the["canh_bao_bien_do_vao_lenh"],
            "ghi_chu": "Chưa bán nên chưa trừ phí bán/thuế bán — PnL tạm tính ở trên là GROSS.",
        }

    ket_qua_loi_nhuan = tinh_loi_nhuan_rong(trades, von_ban_dau, ty_trong_von_pct)

    so_lenh = len(trades)
    so_lan_thang = sum(1 for t in trades if t["final_pnl_pct"] > 0)
    so_lan_canh_bao_bien_do = sum(
        1 for t in trades if t.get("canh_bao_bien_do_vao_lenh") or t.get("canh_bao_bien_do_ra_lenh")
    )

    return {
        "trades": trades,
        "open_position": open_position,
        "canh_bao_tin_hieu_nguoc": canh_bao_tin_hieu_nguoc,
        "so_lenh_da_dong": so_lenh,
        "so_lan_thang": so_lan_thang,
        "so_lan_thua": so_lenh - so_lan_thang,
        "win_rate_pct": round(so_lan_thang / so_lenh * 100, 1) if so_lenh > 0 else None,
        "so_lan_canh_bao_bien_do": so_lan_canh_bao_bien_do,
        "chi_phi_giao_dich_pct_moi_lenh": round(chi_phi_giao_dich_pct, 3),
        **ket_qua_loi_nhuan,
    }


# ==============================================================================
# 4B. KẾT HỢP NHIỀU BỘ THAM SỐ THEO LOGIC OR (bổ sung 06/08/2026) — vào
#     lệnh LONG nếu mã đạt tiêu chí của BẤT KỲ 1 trong N bộ tham số đã
#     chọn (VD: kết hợp Top 10 bộ tốt nhất từ grid search) — tăng số
#     lượng tín hiệu, đồng thời giữ nguyên toàn bộ nguyên tắc quản lý
#     vị thế/SL/Trailing TP/phí/thuế/khóa T+/biên độ đã có.
# ==============================================================================

def chay_backtest_ket_hop_nhieu_bo(
    df_ohlcv: pd.DataFrame,
    danh_sach_tham_so: list[dict],
    bo_loc: dict,
    tp_tiers: list[dict],
    von_ban_dau: float = 1_000_000_000,
    ty_trong_von_pct: float = 50.0,
    so_phien_khoa_toi_thieu: int = SO_PHIEN_KHOA_TOI_THIEU_MAC_DINH,
    phi_moi_gioi_pct: float = PHI_MOI_GIOI_PCT_MAC_DINH,
    thue_ban_pct: float = THUE_BAN_PCT_MAC_DINH,
    bien_do_dao_dong_pct: float = BIEN_DO_DAO_DONG_PCT_MAC_DINH,
) -> dict:
    """Giống hệt `chay_backtest()` (cùng nguyên tắc LONG-only, khóa T+,
    phí/thuế, cảnh báo biên độ, làm tròn lô) — CHỈ khác cách đánh giá
    tín hiệu: dùng `danh_gia_tin_hieu_ket_hop()` để vào lệnh nếu ĐẠT
    tiêu chí của BẤT KỲ bộ nào trong `danh_sach_tham_so` (logic OR).

    `danh_sach_tham_so`: list các dict tham số (mỗi dict cùng cấu trúc
    như tham số `tham_so` của `chay_backtest()`) — VD kết hợp Top 10 bộ
    tốt nhất tìm được từ `experimental/grid_search_lab.py`.
    """
    _validate_df(df_ohlcv)
    _validate_tp_tiers(tp_tiers)
    if not danh_sach_tham_so:
        raise InvalidIndicatorLabError("danh_sach_tham_so không được rỗng.")
    if von_ban_dau <= 0:
        raise InvalidIndicatorLabError("von_ban_dau phải > 0.")
    if not (0 < ty_trong_von_pct <= 100):
        raise InvalidIndicatorLabError("ty_trong_von_pct phải trong khoảng (0, 100].")
    if so_phien_khoa_toi_thieu < 0:
        raise InvalidIndicatorLabError("so_phien_khoa_toi_thieu phải >= 0.")
    if phi_moi_gioi_pct < 0 or thue_ban_pct < 0:
        raise InvalidIndicatorLabError("phi_moi_gioi_pct và thue_ban_pct phải >= 0.")
    if bien_do_dao_dong_pct <= 0:
        raise InvalidIndicatorLabError("bien_do_dao_dong_pct phải > 0.")

    chi_phi_giao_dich_pct = phi_moi_gioi_pct * 2 + thue_ban_pct

    df = df_ohlcv.reset_index(drop=True)
    # Tính chỉ báo RIÊNG cho TỪNG bộ tham số (vì chu kỳ EMA/MA có thể khác
    # nhau giữa các bộ) — chỉ tính 1 lần cho mỗi bộ trước khi vào vòng lặp.
    danh_sach_chi_bao = [tinh_toan_chi_bao(df, ts, bo_loc) for ts in danh_sach_tham_so]
    n_rows = len(df)
    start_idx = max(_so_phien_khoi_dong_toi_thieu(ts, bo_loc) for ts in danh_sach_tham_so)

    trades: list[dict] = []
    canh_bao_tin_hieu_nguoc: list[dict] = []
    vi_the: Optional[dict] = None

    for i in range(start_idx, n_rows):
        close_i = float(df["close"].iloc[i])

        if vi_the is not None:
            da_du_khoa_T = (i - vi_the["entry_idx"]) >= so_phien_khoa_toi_thieu

            if not da_du_khoa_T:
                pass
            else:
                if close_i < vi_the["body_mid"]:
                    pnl_pct = _tinh_pnl_hien_tai_pct(vi_the, close_i)
                    trades.append(_hoan_tat_dong_lenh(
                        vi_the, i, df, pnl_pct, chi_phi_giao_dich_pct, bien_do_dao_dong_pct,
                    ))
                    vi_the = None
                else:
                    pnl_hien_tai = _tinh_pnl_hien_tai_pct(vi_the, close_i)
                    for idx_tier, tier in enumerate(tp_tiers):
                        if vi_the["tiers_kich_hoat"][idx_tier]:
                            continue
                        if pnl_hien_tai >= tier["muc_lai_pct"]:
                            vi_the["tiers_kich_hoat"][idx_tier] = True
                            pct_chot = tier["chot_pct_khoi_luong"]
                            vi_the["pnl_tich_luy_co_trong_so"] += pct_chot * pnl_hien_tai
                            vi_the["remaining_pct"] -= pct_chot
                            if vi_the["remaining_pct"] <= 1e-9:
                                final_pnl_pct_truoc_phi = vi_the["pnl_tich_luy_co_trong_so"] / 100
                                final_pnl_pct = round(final_pnl_pct_truoc_phi - chi_phi_giao_dich_pct, 2)
                                exit_price = vi_the["entry_price"] * (1 + final_pnl_pct / 100)
                                gia_tham_chieu_hom_truoc = float(df["close"].iloc[i - 1]) if i > 0 else None
                                trades.append({
                                    "side": "LONG", "entry_date": vi_the["entry_date"],
                                    "entry_price": round(vi_the["entry_price"], 2),
                                    "exit_date": str(df["date"].iloc[i]), "exit_price": round(exit_price, 2),
                                    "final_pnl_pct": final_pnl_pct,
                                    "so_co_phieu_uoc_tinh": vi_the["so_co_phieu_uoc_tinh"],
                                    "canh_bao_bien_do_vao_lenh": vi_the["canh_bao_bien_do_vao_lenh"],
                                    "canh_bao_bien_do_ra_lenh": _kiem_tra_gan_bien_do(close_i, gia_tham_chieu_hom_truoc, bien_do_dao_dong_pct),
                                })
                                vi_the = None
                                break

        if vi_the is None:
            tin_hieu = danh_gia_tin_hieu_ket_hop(df, i, danh_sach_tham_so, bo_loc, danh_sach_chi_bao)
            if tin_hieu == "BUY":
                vi_the = _mo_lenh_moi(
                    "LONG", i, df, len(tp_tiers), von_ban_dau, ty_trong_von_pct, bien_do_dao_dong_pct,
                )
        else:
            tin_hieu = danh_gia_tin_hieu_ket_hop(df, i, danh_sach_tham_so, bo_loc, danh_sach_chi_bao)
            if tin_hieu == "SELL":
                canh_bao_tin_hieu_nguoc.append({
                    "ngay": str(df["date"].iloc[i]), "tin_hieu_nguoc": tin_hieu,
                    "dang_giu_lenh": vi_the["side"],
                })

    open_position = None
    if vi_the is not None:
        close_cuoi = float(df["close"].iloc[-1])
        open_position = {
            "side": vi_the["side"],
            "entry_date": vi_the["entry_date"],
            "entry_price": round(vi_the["entry_price"], 2),
            "as_of_date": str(df["date"].iloc[-1]),
            "current_price": round(close_cuoi, 2),
            "unrealized_pnl_pct": round(_tinh_pnl_hien_tai_pct(vi_the, close_cuoi), 2),
            "so_co_phieu_uoc_tinh": vi_the["so_co_phieu_uoc_tinh"],
            "canh_bao_bien_do_vao_lenh": vi_the["canh_bao_bien_do_vao_lenh"],
            "ghi_chu": "Chưa bán nên chưa trừ phí bán/thuế bán — PnL tạm tính ở trên là GROSS.",
        }

    ket_qua_loi_nhuan = tinh_loi_nhuan_rong(trades, von_ban_dau, ty_trong_von_pct)

    so_lenh = len(trades)
    so_lan_thang = sum(1 for t in trades if t["final_pnl_pct"] > 0)
    so_lan_canh_bao_bien_do = sum(
        1 for t in trades if t.get("canh_bao_bien_do_vao_lenh") or t.get("canh_bao_bien_do_ra_lenh")
    )

    return {
        "trades": trades,
        "open_position": open_position,
        "canh_bao_tin_hieu_nguoc": canh_bao_tin_hieu_nguoc,
        "so_lenh_da_dong": so_lenh,
        "so_lan_thang": so_lan_thang,
        "so_lan_thua": so_lenh - so_lan_thang,
        "win_rate_pct": round(so_lan_thang / so_lenh * 100, 1) if so_lenh > 0 else None,
        "so_lan_canh_bao_bien_do": so_lan_canh_bao_bien_do,
        "chi_phi_giao_dich_pct_moi_lenh": round(chi_phi_giao_dich_pct, 3),
        "so_bo_tham_so_ket_hop": len(danh_sach_tham_so),
        **ket_qua_loi_nhuan,
    }


# ==============================================================================
# 4C. DANH MỤC NHIỀU MÃ — MỖI MÃ 1 BỘ TIÊU CHÍ RIÊNG (bổ sung 06/08/2026)
#     — cho phép GIỮ ĐỒNG THỜI nhiều vị thế (mỗi mã 1 vị thế riêng, dùng
#     CHUNG 1 quỹ vốn) — KHÁC với chay_backtest_ket_hop_nhieu_bo() (đó
#     là NHIỀU BỘ tiêu chí cho CÙNG 1 mã, chỉ giữ 1 vị thế tại 1 thời điểm).
# ==============================================================================

def chay_backtest_nhieu_ma(
    danh_sach_ma: list[dict],
    bo_loc: dict,
    tp_tiers: list[dict],
    von_ban_dau: float = 1_000_000_000,
    max_tong_von_su_dung_pct: float = 80.0,
    so_phien_khoa_toi_thieu: int = SO_PHIEN_KHOA_TOI_THIEU_MAC_DINH,
    phi_moi_gioi_pct: float = PHI_MOI_GIOI_PCT_MAC_DINH,
    thue_ban_pct: float = THUE_BAN_PCT_MAC_DINH,
    bien_do_dao_dong_pct: float = BIEN_DO_DAO_DONG_PCT_MAC_DINH,
) -> dict:
    """Chạy backtest ĐỒNG THỜI cho NHIỀU MÃ — mỗi mã dùng ĐÚNG bộ tiêu
    chí (tham_so) RIÊNG của nó, và có thể GIỮ ĐỒNG THỜI vị thế ở nhiều
    mã cùng lúc (khác hẳn `chay_backtest()`/`chay_backtest_ket_hop_nhieu_bo()`
    — cả 2 hàm đó chỉ giữ ĐÚNG 1 vị thế tại 1 thời điểm).

    `danh_sach_ma`: list các dict {"ma": str, "df": pd.DataFrame, "tham_so": dict}
    — mỗi phần tử là 1 mã kèm dữ liệu OHLCV VÀ bộ tham số RIÊNG của mã đó
    (VD tham số đã tìm được tốt nhất cho từng mã qua grid search riêng biệt).

    PHÂN BỔ VỐN — "chia đều, không bao giờ vượt trần" (đã thống nhất
    06/08/2026): mỗi mã dùng CỐ ĐỊNH `max_tong_von_su_dung_pct / số mã`
    % vốn cho MỌI lệnh của nó — đảm bảo TOÁN HỌC rằng dù bao nhiêu mã
    cùng mở vị thế lúc nào, tổng % vốn sử dụng KHÔNG BAO GIỜ vượt
    `max_tong_von_su_dung_pct` (vì có đúng N mã, mỗi mã tối đa max/N%).

    `equity` (vốn chung) được cập nhật THEO ĐÚNG THỨ TỰ THỜI GIAN THẬT
    (không phải theo thứ tự trong danh sách mã) — vì các vị thế có thể
    CHỒNG LẤN thời gian giữa các mã khác nhau, không thể dùng lại cách
    "compound tuần tự theo trades list" như hàm đơn-mã.
    """
    if not danh_sach_ma:
        raise InvalidIndicatorLabError("danh_sach_ma không được rỗng.")
    _validate_tp_tiers(tp_tiers)
    if von_ban_dau <= 0:
        raise InvalidIndicatorLabError("von_ban_dau phải > 0.")
    if not (0 < max_tong_von_su_dung_pct <= 100):
        raise InvalidIndicatorLabError("max_tong_von_su_dung_pct phải trong khoảng (0, 100].")
    if so_phien_khoa_toi_thieu < 0:
        raise InvalidIndicatorLabError("so_phien_khoa_toi_thieu phải >= 0.")
    if phi_moi_gioi_pct < 0 or thue_ban_pct < 0:
        raise InvalidIndicatorLabError("phi_moi_gioi_pct và thue_ban_pct phải >= 0.")
    if bien_do_dao_dong_pct <= 0:
        raise InvalidIndicatorLabError("bien_do_dao_dong_pct phải > 0.")

    chi_phi_giao_dich_pct = phi_moi_gioi_pct * 2 + thue_ban_pct
    so_luong_ma = len(danh_sach_ma)
    ty_trong_von_pct_moi_ma = max_tong_von_su_dung_pct / so_luong_ma

    # --- Chuẩn bị: tính chỉ báo 1 lần/mã, dựng map ngày -> vị trí dòng ---
    ma_states: dict[str, dict] = {}
    for muc in danh_sach_ma:
        ma = muc["ma"]
        df = muc["df"].reset_index(drop=True)
        _validate_df(df)
        tham_so = muc["tham_so"]
        chi_bao = tinh_toan_chi_bao(df, tham_so, bo_loc)
        ma_states[ma] = {
            "df": df,
            "tham_so": tham_so,
            "chi_bao": chi_bao,
            "date_to_idx": {d: i for i, d in enumerate(df["date"])},
            "start_idx": _so_phien_khoi_dong_toi_thieu(tham_so, bo_loc),
            "vi_the": None,
            "trades": [],
        }

    tat_ca_ngay = sorted(set().union(*[set(s["df"]["date"]) for s in ma_states.values()]))

    equity = von_ban_dau
    canh_bao_tin_hieu_nguoc: list[dict] = []

    for ngay in tat_ca_ngay:
        for ma, s in ma_states.items():
            idx = s["date_to_idx"].get(ngay)
            if idx is None or idx < s["start_idx"]:
                continue  # mã này không có dữ liệu ngày này, hoặc chưa đủ nền

            df, chi_bao, tham_so = s["df"], s["chi_bao"], s["tham_so"]
            close_i = float(df["close"].iloc[idx])
            vi_the = s["vi_the"]

            if vi_the is not None:
                da_du_khoa_T = (idx - vi_the["entry_idx"]) >= so_phien_khoa_toi_thieu
                if da_du_khoa_T:
                    if close_i < vi_the["body_mid"]:
                        pnl_pct = _tinh_pnl_hien_tai_pct(vi_the, close_i)
                        trade_info = _hoan_tat_dong_lenh(
                            vi_the, idx, df, pnl_pct, chi_phi_giao_dich_pct, bien_do_dao_dong_pct,
                        )
                        equity += vi_the["equity_luc_mo_lenh"] * (ty_trong_von_pct_moi_ma / 100) * (trade_info["final_pnl_pct"] / 100)
                        trade_info["ma"] = ma
                        s["trades"].append(trade_info)
                        s["vi_the"] = None
                        vi_the = None
                    else:
                        pnl_hien_tai = _tinh_pnl_hien_tai_pct(vi_the, close_i)
                        for idx_tier, tier in enumerate(tp_tiers):
                            if vi_the["tiers_kich_hoat"][idx_tier]:
                                continue
                            if pnl_hien_tai >= tier["muc_lai_pct"]:
                                vi_the["tiers_kich_hoat"][idx_tier] = True
                                pct_chot = tier["chot_pct_khoi_luong"]
                                vi_the["pnl_tich_luy_co_trong_so"] += pct_chot * pnl_hien_tai
                                vi_the["remaining_pct"] -= pct_chot
                                if vi_the["remaining_pct"] <= 1e-9:
                                    final_pnl_pct_truoc_phi = vi_the["pnl_tich_luy_co_trong_so"] / 100
                                    final_pnl_pct = round(final_pnl_pct_truoc_phi - chi_phi_giao_dich_pct, 2)
                                    exit_price = vi_the["entry_price"] * (1 + final_pnl_pct / 100)
                                    gia_tham_chieu_hom_truoc = float(df["close"].iloc[idx - 1]) if idx > 0 else None
                                    equity += vi_the["equity_luc_mo_lenh"] * (ty_trong_von_pct_moi_ma / 100) * (final_pnl_pct / 100)
                                    s["trades"].append({
                                        "ma": ma, "side": "LONG", "entry_date": vi_the["entry_date"],
                                        "entry_price": round(vi_the["entry_price"], 2),
                                        "exit_date": str(df["date"].iloc[idx]), "exit_price": round(exit_price, 2),
                                        "final_pnl_pct": final_pnl_pct,
                                        "so_co_phieu_uoc_tinh": vi_the["so_co_phieu_uoc_tinh"],
                                        "canh_bao_bien_do_vao_lenh": vi_the["canh_bao_bien_do_vao_lenh"],
                                        "canh_bao_bien_do_ra_lenh": _kiem_tra_gan_bien_do(close_i, gia_tham_chieu_hom_truoc, bien_do_dao_dong_pct),
                                    })
                                    s["vi_the"] = None
                                    vi_the = None
                                    break

            if s["vi_the"] is None:
                tin_hieu = danh_gia_tin_hieu(df, idx, tham_so, bo_loc, chi_bao)
                if tin_hieu == "BUY":
                    vi_the_moi = _mo_lenh_moi(
                        "LONG", idx, df, len(tp_tiers), von_ban_dau, ty_trong_von_pct_moi_ma, bien_do_dao_dong_pct,
                    )
                    vi_the_moi["equity_luc_mo_lenh"] = equity  # LƯU equity TẠI THỜI ĐIỂM MỞ LỆNH (để compound đúng lúc đóng)
                    s["vi_the"] = vi_the_moi
            else:
                tin_hieu = danh_gia_tin_hieu(df, idx, tham_so, bo_loc, chi_bao)
                if tin_hieu == "SELL":
                    canh_bao_tin_hieu_nguoc.append({
                        "ma": ma, "ngay": str(ngay), "tin_hieu_nguoc": tin_hieu,
                    })

    open_positions_theo_ma: dict[str, Optional[dict]] = {}
    for ma, s in ma_states.items():
        vi_the = s["vi_the"]
        if vi_the is None:
            open_positions_theo_ma[ma] = None
            continue
        df = s["df"]
        close_cuoi = float(df["close"].iloc[-1])
        open_positions_theo_ma[ma] = {
            "side": vi_the["side"], "entry_date": vi_the["entry_date"],
            "entry_price": round(vi_the["entry_price"], 2),
            "as_of_date": str(df["date"].iloc[-1]), "current_price": round(close_cuoi, 2),
            "unrealized_pnl_pct": round(_tinh_pnl_hien_tai_pct(vi_the, close_cuoi), 2),
            "so_co_phieu_uoc_tinh": vi_the["so_co_phieu_uoc_tinh"],
            "canh_bao_bien_do_vao_lenh": vi_the["canh_bao_bien_do_vao_lenh"],
        }

    trades_theo_ma = {ma: s["trades"] for ma, s in ma_states.items()}
    tat_ca_trades = [t for trades in trades_theo_ma.values() for t in trades]
    tong_so_lenh = len(tat_ca_trades)
    tong_so_lan_thang = sum(1 for t in tat_ca_trades if t["final_pnl_pct"] > 0)

    return {
        "trades_theo_ma": trades_theo_ma,
        "open_positions_theo_ma": open_positions_theo_ma,
        "canh_bao_tin_hieu_nguoc": canh_bao_tin_hieu_nguoc,
        "tong_so_lenh_da_dong": tong_so_lenh,
        "tong_so_lan_thang": tong_so_lan_thang,
        "tong_so_lan_thua": tong_so_lenh - tong_so_lan_thang,
        "win_rate_pct": round(tong_so_lan_thang / tong_so_lenh * 100, 1) if tong_so_lenh > 0 else None,
        "von_ban_dau": von_ban_dau,
        "von_cuoi_cung": round(equity),
        "loi_nhuan_rong": round(equity - von_ban_dau),
        "loi_nhuan_rong_pct": round((equity - von_ban_dau) / von_ban_dau * 100, 2) if von_ban_dau > 0 else None,
        "ty_trong_von_pct_moi_ma": round(ty_trong_von_pct_moi_ma, 2),
        "so_luong_ma_ket_hop": so_luong_ma,
        "chi_phi_giao_dich_pct_moi_lenh": round(chi_phi_giao_dich_pct, 3),
    }


# ==============================================================================
# 5. CÔNG THỨC LỢI NHUẬN RÒNG (compound theo đúng thứ tự thời gian)
# ==============================================================================

def tinh_loi_nhuan_rong(trades: list[dict], von_ban_dau: float, ty_trong_von_pct: float) -> dict:
    equity = von_ban_dau
    for t in trades:
        equity += equity * (ty_trong_von_pct / 100) * (t["final_pnl_pct"] / 100)

    return {
        "von_ban_dau": von_ban_dau,
        "von_cuoi_cung": round(equity),
        "loi_nhuan_rong": round(equity - von_ban_dau),
        "loi_nhuan_rong_pct": (
            round((equity - von_ban_dau) / von_ban_dau * 100, 2) if von_ban_dau > 0 else None
        ),
    }


# ==============================================================================
# 6. QUÉT TOÀN BỘ WATCHLIST — tìm danh sách mã đang thỏa BUY/SELL tại
#    ĐÚNG NẾN CUỐI CÙNG, dùng CHÍNH bộ tham số/bộ lọc đang cấu hình.
#    GIỮ NGUYÊN 100% nguyên tắc lọc ở mục 1+3 — không nới lỏng/thêm bớt
#    điều kiện gì khi quét hàng loạt.
# ==============================================================================

def quet_watchlist_tim_tin_hieu(
    du_lieu_theo_ma: dict[str, pd.DataFrame], tham_so: dict, bo_loc: dict,
    nguong_volume_ma20_toi_thieu: Optional[float] = 300_000,
    muc_chot_loi_pct: tuple[float, ...] = (5.0, 10.0, 15.0),
) -> dict:
    """Với mỗi mã trong `du_lieu_theo_ma`, chỉ kiểm tra ĐÚNG NẾN CUỐI CÙNG
    hiện có xem có thỏa mãn đầy đủ điều kiện BUY hoặc SELL hay không.

    `nguong_volume_ma20_toi_thieu` (bổ sung 06/08/2026): LOẠI TRỪ mã có
    khối lượng TB 20 phiên (MA20 — LUÔN tính cố định chu kỳ 20, độc lập
    với bộ lọc Volume tùy chọn ở `bo_loc` có thể dùng chu kỳ khác) dưới
    ngưỡng này — tránh gợi ý mã quá thanh khoản thấp. Để `None` để tắt.

    `muc_chot_loi_pct` (bổ sung 06/08/2026): các mốc % để tính GIÁ CHỐT
    LỜI gợi ý (mặc định 5%/10%/15%, khớp đúng 3 tier Trailing TP mặc
    định) — chỉ mang tính THAM KHẢO, KHÔNG phải lệnh tự động.

    Trả về {"buy": [...], "sell": [...]} — mỗi phần tử là 1 dict thông
    tin gọn cho 1 mã (không chạy backtest đầy đủ, chỉ đánh giá 1 điểm).
    """
    ket_qua_buy: list[dict] = []
    ket_qua_sell: list[dict] = []

    for ma, df in du_lieu_theo_ma.items():
        if df is None or df.empty:
            continue
        try:
            _validate_df(df)
        except InvalidIndicatorLabError:
            continue

        so_phien_can = max(_so_phien_khoi_dong_toi_thieu(tham_so, bo_loc), 20)
        if len(df) <= so_phien_can:
            continue

        df_reset = df.reset_index(drop=True)
        try:
            chi_bao = tinh_toan_chi_bao(df_reset, tham_so, bo_loc)
        except Exception:  # noqa: BLE001
            continue

        i = len(df_reset) - 1

        # --- Bộ lọc loại trừ thanh khoản thấp (LUÔN tính MA20 cố định,
        #     độc lập với bộ lọc Volume tùy chọn ở bo_loc) ---
        volume_ma20_series = calculate_volume_ma(df_reset, period=20)
        volume_ma20_i = volume_ma20_series.iloc[i]
        if nguong_volume_ma20_toi_thieu is not None:
            if pd.isna(volume_ma20_i) or volume_ma20_i < nguong_volume_ma20_toi_thieu:
                continue

        tin_hieu = danh_gia_tin_hieu(df_reset, i, tham_so, bo_loc, chi_bao)
        if tin_hieu is None:
            continue

        gia_dong_cua = float(df_reset["close"].iloc[i])
        body_mid = float((df_reset["high"].iloc[i] + df_reset["low"].iloc[i]) / 2)

        hang = {
            "ma": ma,
            "ngay_tin_hieu": str(df_reset["date"].iloc[i]),
            "gia_dong_cua": gia_dong_cua,
            "body_pct": candle_body_pct(df_reset, i),
            "gia_de_nghi_vao_lenh": round(gia_dong_cua, 2),
            "gia_cutloss": round(body_mid, 2),
            "volume": float(df_reset["volume"].iloc[i]),
            "volume_ma20": round(float(volume_ma20_i)) if not pd.isna(volume_ma20_i) else None,
        }
        for muc_pct in muc_chot_loi_pct:
            hang[f"gia_chot_loi_{muc_pct:g}pct"] = round(gia_dong_cua * (1 + muc_pct / 100), 2)

        if bo_loc.get("rsi_enabled"):
            rsi_i = chi_bao["rsi"].iloc[i]
            hang["rsi"] = round(float(rsi_i), 1) if not pd.isna(rsi_i) else None
        if bo_loc.get("atr_enabled"):
            atr_i = chi_bao["atr"].iloc[i]
            hang["atr"] = round(float(atr_i), 2) if not pd.isna(atr_i) else None

        if tin_hieu == "BUY":
            ket_qua_buy.append(hang)
        else:
            ket_qua_sell.append(hang)

    return {"buy": ket_qua_buy, "sell": ket_qua_sell}
