"""
experimental/portfolio_backtest_lab.py
==========================================
Script chạy backtest DANH MỤC NHIỀU MÃ — mỗi mã dùng ĐÚNG bộ tham số
RIÊNG của nó (VD tìm được qua `experimental/grid_search_lab.py` cho
từng mã), GIỮ ĐỒNG THỜI được nhiều vị thế (mỗi mã 1 vị thế riêng, dùng
CHUNG 1 quỹ vốn — chia đều theo số mã, không bao giờ vượt trần đã đặt).

CÁCH DÙNG:
    1. Chạy `grid_search_lab.py` riêng cho TỪNG mã bạn muốn kết hợp, ghi
       lại bộ tham số tốt nhất của mỗi mã.
    2. Sửa DANH_MUC bên dưới — điền đúng bộ tham số của TỪNG mã.
    3. Chạy:
        set PM_CK_DB_PATH=postgresql://...
        python -m experimental.portfolio_backtest_lab

⚠️ CẢNH BÁO OVERFIT: các bộ tham số trong danh mục thường đã được TỐI ƯU
RIÊNG cho từng mã trên CÙNG 1 giai đoạn lịch sử — kết hợp lại KHÔNG đảm
bảo hiệu suất tương lai, chỉ mang tính THAM KHẢO.
"""

from __future__ import annotations

import pandas as pd

from experimental.indicator_lab import BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH, chay_backtest_nhieu_ma
from main import load_config, resolve_storage_path
from core.storage import Storage

# ==============================================================================
# DANH MỤC — điền đúng bộ tham số TỐT NHẤT của TỪNG mã (tìm qua grid_search_lab.py
# chạy riêng cho từng mã). Ví dụ minh họa dưới đây dùng cùng 1 bộ cho cả
# 2 mã — SỬA LẠI cho đúng kết quả grid search thật của bạn.
# ==============================================================================

DANH_MUC = [
    {
        "ma": "HDB",
        "tham_so": {
            "buy_lookback": 3, "sell_lookback": 3, "ema_period": 100, "ma_period": 10,
            "range_pct_max": 10.0, "body_pct_min": 0.5, "body_pct_max": 10.0,
        },
    },
    {
        "ma": "VCB",
        "tham_so": {
            "buy_lookback": 3, "sell_lookback": 3, "ema_period": 100, "ma_period": 10,
            "range_pct_max": 10.0, "body_pct_min": 0.5, "body_pct_max": 10.0,
        },
    },
]

VON_BAN_DAU = 1_000_000_000
MAX_TONG_VON_SU_DUNG_PCT = 80.0  # tổng % vốn tối đa dùng cùng lúc, chia đều cho các mã


def _load_ohlcv(storage: Storage, symbol: str) -> pd.DataFrame:
    record = storage.get_latest("ohlcv_history", symbol)
    if record is None:
        raise SystemExit(f"Không có dữ liệu OHLCV cho mã '{symbol}'.")
    records = record["data"].get("records", [])
    if not records:
        raise SystemExit(f"Dữ liệu OHLCV cho mã '{symbol}' rỗng.")
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    config = load_config()
    storage = Storage(db_path=resolve_storage_path(config))

    danh_sach_ma = []
    for muc in DANH_MUC:
        df = _load_ohlcv(storage, muc["ma"].upper())
        print(f"Đã tải {len(df)} phiên OHLCV cho {muc['ma'].upper()} "
              f"({df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()})")
        danh_sach_ma.append({"ma": muc["ma"].upper(), "df": df, "tham_so": muc["tham_so"]})
    storage.close()

    print(
        "\n⚠️ CẢNH BÁO OVERFIT: các bộ tham số trong danh mục thường đã tối ưu riêng cho "
        "từng mã trên CÙNG 1 giai đoạn lịch sử — kết hợp lại KHÔNG đảm bảo hiệu suất "
        "tương lai, chỉ mang tính THAM KHẢO.\n"
    )

    kq = chay_backtest_nhieu_ma(
        danh_sach_ma, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
        von_ban_dau=VON_BAN_DAU, max_tong_von_su_dung_pct=MAX_TONG_VON_SU_DUNG_PCT,
    )

    print(f"=== KẾT QUẢ DANH MỤC {kq['so_luong_ma_ket_hop']} MÃ (vốn {VON_BAN_DAU:,.0f}đ) ===")
    print(f"Tỷ trọng vốn/mã: {kq['ty_trong_von_pct_moi_ma']}% (= {MAX_TONG_VON_SU_DUNG_PCT}% / {kq['so_luong_ma_ket_hop']} mã)")
    print(f"Tổng số lệnh đã đóng: {kq['tong_so_lenh_da_dong']}")
    print(f"Win rate: {kq['win_rate_pct']}%")
    print(f"Vốn cuối cùng: {kq['von_cuoi_cung']:,.0f}đ")
    print(f"Lợi nhuận ròng: {kq['loi_nhuan_rong']:,.0f}đ ({kq['loi_nhuan_rong_pct']:+.2f}%)")

    print("\n--- Chi tiết theo từng mã ---")
    for ma, trades in kq["trades_theo_ma"].items():
        so_thang = sum(1 for t in trades if t["final_pnl_pct"] > 0)
        op = kq["open_positions_theo_ma"].get(ma)
        dong_mo = " (còn 1 vị thế đang mở)" if op else ""
        print(f"  {ma}: {len(trades)} lệnh ({so_thang} thắng){dong_mo}")

    if kq["canh_bao_tin_hieu_nguoc"]:
        print(f"\n{len(kq['canh_bao_tin_hieu_nguoc'])} lần có tín hiệu SELL cảnh báo khi đang giữ lệnh (không tự đóng).")


if __name__ == "__main__":
    main()
