"""
multi_strategy_year_regime_backtest.py
========================================
[Bổ sung 07/09/2026 — Lọc bộ chỉ số/tổ hợp theo NĂM + GIAI ĐOẠN]

Mở rộng `core/long_term_indicator_backtest.py`: thay vì CHỈ bucket kết
quả theo giai đoạn (Uptrend/Sideway/Downtrend) trên TOÀN BỘ lịch sử,
module này bucket THÊM theo TỪNG NĂM DƯƠNG LỊCH riêng biệt (lãi cộng dồn
TRONG 1 năm cụ thể, không cộng dồn qua các năm khác) — VÀ mở rộng từ 8
bộ chỉ số đơn lẻ sang THÊM 21 TỔ HỢP CẶP (kết hợp 2 trong 7 bộ có điều
kiện vào/ra rõ ràng, không gồm "Mua và giữ" — tổ hợp với Buy&Hold vô
nghĩa vì Buy&Hold không có điều kiện ra).

ĐỊNH NGHĨA TỔ HỢP 2 BỘ CHỈ SỐ: entry = A.entry AND B.entry (cả 2 cùng
đồng ý vào lệnh — chọn lọc hơn 1 bộ đơn lẻ), exit = A.exit OR B.exit
(thoát ngay khi 1 TRONG 2 báo hiệu ra — ưu tiên bảo toàn vốn).

Dùng lại NGUYÊN VẸN `backtest.backtest_engine.run_backtest()` (không
lookahead bias — thực thi ở giá MỞ CỬA ngày kế tiếp) và
`core.long_term_indicator_backtest.xay_8_bo_chi_so()` (không tự viết lại
công thức tín hiệu của 8 bộ đã có).

QUAN TRỌNG: module này KHÔNG tự tính chuỗi giai đoạn (Ensemble 3 phương
pháp cần fit Markov, ~26 giây/mã — RẤT TỐN) — `regime_series` phải được
TRUYỀN VÀO từ bên ngoài, TÁI SỬ DỤNG đúng chuỗi đã tính 1 lần cho mục
"🧮 Cổ phiếu dài hạn" (`main.run_long_term_screener_step()`), tránh tính
lại 2 lần cho CÙNG 1 mã trong CÙNG 1 lượt chạy pipeline.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd

from backtest.backtest_engine import run_backtest
from core.long_term_indicator_backtest import tinh_chi_bao_dai_han, xay_8_bo_chi_so

TEN_KHONG_TO_HOP = "Mua và giữ (Buy & Hold)"


def xay_to_hop_cap_bo_chi_so(
    bo_8: dict[str, tuple[pd.Series, pd.Series]],
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Trả về `{"A + B": (entry, exit)}` cho TẤT CẢ cặp 2 bộ chỉ số trong
    `bo_8` (thường lấy từ `xay_8_bo_chi_so()`), TRỪ `TEN_KHONG_TO_HOP`
    ("Mua và giữ") — entry = A.entry AND B.entry, exit = A.exit OR B.exit.

    Với 7 bộ còn lại (loại Buy&Hold), trả về C(7,2) = 21 tổ hợp.
    """
    ten_hop_le = [t for t in bo_8 if t != TEN_KHONG_TO_HOP]
    ket_qua: dict[str, tuple[pd.Series, pd.Series]] = {}
    for i in range(len(ten_hop_le)):
        for j in range(i + 1, len(ten_hop_le)):
            ten_a, ten_b = ten_hop_le[i], ten_hop_le[j]
            entry_a, exit_a = bo_8[ten_a]
            entry_b, exit_b = bo_8[ten_b]
            ket_qua[f"{ten_a} + {ten_b}"] = (entry_a & entry_b, exit_a | exit_b)
    return ket_qua


def _tim_giai_doan_tai_ngay(ngay, regime_series: pd.Series) -> Optional[str]:
    """Tìm giai đoạn có hiệu lực tại `ngay` trong `regime_series` (index =
    ngày) — dùng ĐÚNG ngày nếu có, hoặc ngày GẦN NHẤT TRƯỚC ĐÓ nếu không
    (chuỗi giai đoạn có thể thưa hơn lịch giao dịch thật). Trả về `None`
    nếu `ngay` sớm hơn toàn bộ chuỗi (chưa đủ dữ liệu tính giai đoạn)."""
    if len(regime_series) == 0:
        return None
    ts = pd.Timestamp(ngay)
    if ts in regime_series.index:
        return regime_series.loc[ts]
    truoc = regime_series.index[regime_series.index <= ts]
    if len(truoc) == 0:
        return None
    return regime_series.loc[truoc[-1]]


def _tong_hop_von_tuan_tu(trades: list, initial_capital: float) -> dict:
    """Mô phỏng vốn TUẦN TỰ (compounding) chỉ với `trades` đã cho — coi
    như đây là TOÀN BỘ lịch sử giao dịch trong phạm vi đang xét (VD 1
    năm cụ thể), bắt đầu lại từ `initial_capital`."""
    von = initial_capital
    thang = 0
    for t in trades:
        von *= (1 + t.pnl_pct / 100.0)
        if t.pnl > 0:
            thang += 1
    return {
        "n_trades": len(trades),
        "win_rate_pct": thang / len(trades) * 100.0,
        "total_return_pct": (von / initial_capital - 1.0) * 100.0,
    }


def backtest_to_hop_theo_nam_giai_doan(
    df: pd.DataFrame,
    regime_series: pd.Series,
    initial_capital: float = 1_000_000_000.0,
    fee_pct: float = 0.15,
    stop_loss_pct: Optional[float] = None,
) -> list[dict]:
    """Backtest 8 bộ chỉ số đơn lẻ + 21 tổ hợp cặp cho 1 mã (`df`), bucket
    kết quả theo TỪNG NĂM có ít nhất 1 lệnh, kèm GIAI ĐOẠN CHỦ YẾU (giai
    đoạn xuất hiện nhiều nhất tại ngày VÀO LỆNH của các lệnh trong năm
    đó, tính theo `regime_series` truyền vào).

    `stop_loss_pct`: nếu truyền (VD 2.0 = cắt lỗ tối đa 2%), chuyển thẳng
    xuống `backtest.backtest_engine.run_backtest()` cho MỌI bộ chỉ số/tổ
    hợp — mỗi lệnh sẽ tự động đóng nếu lỗ (theo giá đóng cửa) chạm ngưỡng
    này, bất kể tín hiệu thoát riêng của chiến lược. Mặc định `None` =
    không áp dụng, giữ nguyên hành vi cũ.

    Trả về list các dict — MỖI DÒNG là 1 tổ hợp (tên_bộ_chỉ_số, năm) có
    ít nhất 1 lệnh trong năm đó:
        {"ten_bo_chi_so", "nam", "giai_doan_chinh",
         "so_lenh_giai_doan_chinh", "tong_so_lenh_co_giai_doan",
         "n_trades", "win_rate_pct", "total_return_pct"}

    KHÔNG lọc theo ngưỡng lãi % ở đây — lọc theo tiêu chí là việc của
    `loc_ket_qua_theo_dieu_kien()` (tầng hiển thị/truy vấn), tách biệt
    khỏi tầng tính toán để có thể lọc lại nhiều tiêu chí khác nhau mà
    KHÔNG cần backtest lại.
    """
    df_bt = tinh_chi_bao_dai_han(df)
    bo_8 = xay_8_bo_chi_so(df_bt)
    to_hop = xay_to_hop_cap_bo_chi_so(bo_8)
    tat_ca_cau_hinh: dict[str, tuple[pd.Series, pd.Series]] = {**bo_8, **to_hop}

    ket_qua: list[dict] = []
    for ten, (entry, exit_) in tat_ca_cau_hinh.items():
        if entry.sum() == 0:
            continue
        result = run_backtest(
            df_bt, entry_signal_fn=lambda _df, e=entry: e, exit_signal_fn=lambda _df, x=exit_: x,
            initial_cash=initial_capital, fee_pct=fee_pct,
            stop_loss_pct=stop_loss_pct,
        )
        trades = result.trades
        if not trades:
            continue

        cac_nam = sorted({pd.Timestamp(t.entry_date).year for t in trades})
        for nam in cac_nam:
            trades_nam = [t for t in trades if pd.Timestamp(t.entry_date).year == nam]
            tong_hop = _tong_hop_von_tuan_tu(trades_nam, initial_capital)

            giai_doan_moi_lenh = [
                gd for gd in (_tim_giai_doan_tai_ngay(t.entry_date, regime_series) for t in trades_nam)
                if gd is not None
            ]
            if giai_doan_moi_lenh:
                giai_doan_chinh, so_lan = Counter(giai_doan_moi_lenh).most_common(1)[0]
            else:
                giai_doan_chinh, so_lan = None, 0

            ket_qua.append({
                "ten_bo_chi_so": ten,
                "nam": nam,
                "giai_doan_chinh": giai_doan_chinh,
                "so_lenh_giai_doan_chinh": so_lan,
                "tong_so_lenh_co_giai_doan": len(giai_doan_moi_lenh),
                "n_trades": tong_hop["n_trades"],
                "win_rate_pct": tong_hop["win_rate_pct"],
                "total_return_pct": tong_hop["total_return_pct"],
            })
    return ket_qua


def loc_ket_qua_theo_dieu_kien(
    danh_sach_hang: list[dict],
    nguong_lai_pct: Optional[float] = None,
    ten_bo_chi_so: Optional[str] = None,
    giai_doan: Optional[str] = None,
    nam: Optional[int] = None,
    so_lenh_toi_thieu: int = 1,
) -> list[dict]:
    """Lọc `danh_sach_hang` (mỗi dòng là 1 dict theo đúng định dạng trả
    về từ `backtest_to_hop_theo_nam_giai_doan()`, thường đã gộp thêm khóa
    "ma" ở tầng gọi khi tổng hợp NHIỀU mã) theo các tiêu chí TÙY CHỌN —
    bỏ trống (None) tiêu chí nào thì KHÔNG lọc theo tiêu chí đó.

    `nguong_lai_pct`: chỉ giữ dòng có `total_return_pct` LỚN HƠN (>) giá
    trị này (không phải >=).
    `so_lenh_toi_thieu`: mặc định 1 — không loại dòng nào chỉ vì ít lệnh,
    NHƯNG các dòng chỉ 1-2 lệnh có độ tin cậy thấp, nên hiển thị kèm cột
    "n_trades" để người xem tự đánh giá thay vì lọc cứng.
    """
    ket_qua = []
    for hang in danh_sach_hang:
        if nguong_lai_pct is not None:
            lai = hang.get("total_return_pct")
            if lai is None or lai <= nguong_lai_pct:
                continue
        if ten_bo_chi_so is not None and hang.get("ten_bo_chi_so") != ten_bo_chi_so:
            continue
        if giai_doan is not None and hang.get("giai_doan_chinh") != giai_doan:
            continue
        if nam is not None and hang.get("nam") != nam:
            continue
        if hang.get("n_trades", 0) < so_lenh_toi_thieu:
            continue
        ket_qua.append(hang)
    return ket_qua
