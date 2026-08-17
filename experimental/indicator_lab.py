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

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


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


# ==============================================================================
# 4. ENGINE CHẠY THỬ (BACKTEST) — quản lý vị thế, SL, trailing TP bậc thang
# ==============================================================================

def _mo_lenh_moi(side: str, i: int, df: pd.DataFrame, so_tiers: int) -> dict:
    return {
        "side": side,
        "entry_idx": i,
        "entry_date": str(df["date"].iloc[i]),
        "entry_price": float(df["close"].iloc[i]),
        "body_mid": float((df["high"].iloc[i] + df["low"].iloc[i]) / 2),
        "remaining_pct": 100.0,
        "tiers_kich_hoat": [False] * so_tiers,
        "pnl_tich_luy_co_trong_so": 0.0,
    }


def _tinh_pnl_hien_tai_pct(vi_the: dict, gia_hien_tai: float) -> float:
    if vi_the["side"] == "LONG":
        return (gia_hien_tai - vi_the["entry_price"]) / vi_the["entry_price"] * 100
    return (vi_the["entry_price"] - gia_hien_tai) / vi_the["entry_price"] * 100


def _hoan_tat_dong_lenh(vi_the: dict, i: int, df: pd.DataFrame, pnl_pct_cho_phan_con_lai: float) -> dict:
    vi_the["pnl_tich_luy_co_trong_so"] += vi_the["remaining_pct"] * pnl_pct_cho_phan_con_lai
    final_pnl_pct = round(vi_the["pnl_tich_luy_co_trong_so"] / 100, 2)
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
    }


def chay_backtest(
    df_ohlcv: pd.DataFrame,
    tham_so: dict,
    bo_loc: dict,
    tp_tiers: list[dict],
    von_ban_dau: float = 1_000_000_000,
    ty_trong_von_pct: float = 50.0,
    chi_giao_dich_mot_chieu: Optional[str] = None,
) -> dict:
    """Chạy backtest tuần tự trên TOÀN BỘ `df_ohlcv` theo đúng bộ tham số/
    bộ lọc/trailing TP truyền vào. KHÔNG ghi vào storage — chỉ trả về dict
    kết quả để lớp gọi (dashboard) tự lưu tạm vào `st.session_state`.

    `chi_giao_dich_mot_chieu` (bổ sung 06/08/2026): "LONG" hoặc "SHORT" —
    nếu truyền, CHỈ mở lệnh theo đúng chiều này (tín hiệu chiều ngược lại
    bị bỏ qua hoàn toàn, không mở lệnh, không tính cảnh báo tín hiệu
    ngược). Dùng để tách riêng hiệu suất LONG/SHORT khi dò tham số
    (grid search) — mặc định `None` giữ nguyên hành vi cũ (giao dịch cả
    2 chiều theo đúng tín hiệu).
    """
    _validate_df(df_ohlcv)
    _validate_tp_tiers(tp_tiers)
    if von_ban_dau <= 0:
        raise InvalidIndicatorLabError("von_ban_dau phải > 0.")
    if not (0 < ty_trong_von_pct <= 100):
        raise InvalidIndicatorLabError("ty_trong_von_pct phải trong khoảng (0, 100].")
    if chi_giao_dich_mot_chieu is not None and chi_giao_dich_mot_chieu not in ("LONG", "SHORT"):
        raise InvalidIndicatorLabError('chi_giao_dich_mot_chieu phải là "LONG", "SHORT", hoặc None.')

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
            # --- 1) Kiểm tra Stop Loss trước ---
            da_dong_boi_sl = False
            if vi_the["side"] == "LONG" and close_i < vi_the["body_mid"]:
                da_dong_boi_sl = True
            elif vi_the["side"] == "SHORT" and close_i > vi_the["body_mid"]:
                da_dong_boi_sl = True

            if da_dong_boi_sl:
                pnl_pct = _tinh_pnl_hien_tai_pct(vi_the, close_i)
                trades.append(_hoan_tat_dong_lenh(vi_the, i, df, pnl_pct))
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
                            final_pnl_pct = round(vi_the["pnl_tich_luy_co_trong_so"] / 100, 2)
                            if vi_the["side"] == "LONG":
                                exit_price = vi_the["entry_price"] * (1 + final_pnl_pct / 100)
                            else:
                                exit_price = vi_the["entry_price"] * (1 - final_pnl_pct / 100)
                            trades.append({
                                "side": vi_the["side"], "entry_date": vi_the["entry_date"],
                                "entry_price": round(vi_the["entry_price"], 2),
                                "exit_date": str(df["date"].iloc[i]), "exit_price": round(exit_price, 2),
                                "final_pnl_pct": final_pnl_pct,
                            })
                            vi_the = None
                            break

        if vi_the is None:
            tin_hieu = danh_gia_tin_hieu(df, i, tham_so, bo_loc, chi_bao)
            if tin_hieu == "BUY" and chi_giao_dich_mot_chieu != "SHORT":
                vi_the = _mo_lenh_moi("LONG", i, df, len(tp_tiers))
            elif tin_hieu == "SELL" and chi_giao_dich_mot_chieu != "LONG":
                vi_the = _mo_lenh_moi("SHORT", i, df, len(tp_tiers))
        else:
            # Đang giữ lệnh mà có tín hiệu NGƯỢC CHIỀU -> chỉ ghi log cảnh
            # báo, KHÔNG tự đóng/mở/đảo chiều (đúng yêu cầu prompt).
            tin_hieu = danh_gia_tin_hieu(df, i, tham_so, bo_loc, chi_bao)
            huong_hien_tai = "BUY" if vi_the["side"] == "LONG" else "SELL"
            if tin_hieu is not None and tin_hieu != huong_hien_tai:
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
        }

    ket_qua_loi_nhuan = tinh_loi_nhuan_rong(trades, von_ban_dau, ty_trong_von_pct)

    so_lenh = len(trades)
    so_lan_thang = sum(1 for t in trades if t["final_pnl_pct"] > 0)

    return {
        "trades": trades,
        "open_position": open_position,
        "canh_bao_tin_hieu_nguoc": canh_bao_tin_hieu_nguoc,
        "so_lenh_da_dong": so_lenh,
        "so_lan_thang": so_lan_thang,
        "so_lan_thua": so_lenh - so_lan_thang,
        "win_rate_pct": round(so_lan_thang / so_lenh * 100, 1) if so_lenh > 0 else None,
        **ket_qua_loi_nhuan,
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
    du_lieu_theo_ma: dict[str, pd.DataFrame], tham_so: dict, bo_loc: dict
) -> dict:
    """Với mỗi mã trong `du_lieu_theo_ma`, chỉ kiểm tra ĐÚNG NẾN CUỐI CÙNG
    hiện có xem có thỏa mãn đầy đủ điều kiện BUY hoặc SELL hay không.

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

        so_phien_can = _so_phien_khoi_dong_toi_thieu(tham_so, bo_loc)
        if len(df) <= so_phien_can:
            continue

        df_reset = df.reset_index(drop=True)
        try:
            chi_bao = tinh_toan_chi_bao(df_reset, tham_so, bo_loc)
        except Exception:  # noqa: BLE001
            continue

        i = len(df_reset) - 1
        tin_hieu = danh_gia_tin_hieu(df_reset, i, tham_so, bo_loc, chi_bao)
        if tin_hieu is None:
            continue

        hang = {
            "ma": ma,
            "ngay_tin_hieu": str(df_reset["date"].iloc[i]),
            "gia_dong_cua": float(df_reset["close"].iloc[i]),
            "body_pct": candle_body_pct(df_reset, i),
        }
        if bo_loc.get("rsi_enabled"):
            rsi_i = chi_bao["rsi"].iloc[i]
            hang["rsi"] = round(float(rsi_i), 1) if not pd.isna(rsi_i) else None
        if bo_loc.get("atr_enabled"):
            atr_i = chi_bao["atr"].iloc[i]
            hang["atr"] = round(float(atr_i), 2) if not pd.isna(atr_i) else None
        if bo_loc.get("volume_enabled"):
            hang["volume"] = float(df_reset["volume"].iloc[i])
            vol_ma_i = chi_bao["volume_ma"].iloc[i]
            hang["volume_ma"] = round(float(vol_ma_i)) if not pd.isna(vol_ma_i) else None

        if tin_hieu == "BUY":
            ket_qua_buy.append(hang)
        else:
            ket_qua_sell.append(hang)

    return {"buy": ket_qua_buy, "sell": ket_qua_sell}
