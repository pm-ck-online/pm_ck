"""
experimental/grid_search_lab.py
==================================
Script DÒ THAM SỐ (grid search) — chạy NGOÀI giao diện Streamlit, đúng
theo thiết kế ban đầu của "Phòng thí nghiệm chỉ báo" (mục 10 prompt gốc:
"KHÔNG tự động dò grid search trên UI — dò nhiều tổ hợp làm ở script
riêng ngoài UI nếu cần").

Thử NHIỀU tổ hợp tham số cho 1 mã. Engine (`chay_backtest`) đã tự giới
hạn CHỈ giao dịch LONG (đúng thực tế TTCK Việt Nam — cổ phiếu thường
không bán khống được), nên không cần truyền thêm tham số hướng giao
dịch. Xếp hạng theo lợi nhuận ròng, in ra bảng kết quả Top N.

⚠️ CẢNH BÁO OVERFIT: dò càng nhiều tổ hợp trên CÙNG 1 bộ dữ liệu lịch sử,
càng dễ tìm ra 1 bộ "đẹp trên giấy" nhưng KHÔNG chắc lặp lại tương lai.
Dùng kết quả này để THAM KHẢO thu hẹp phạm vi thử nghiệm tiếp, KHÔNG
phải kết luận cuối cùng.

CÁCH CHẠY:
    set PM_CK_DB_PATH=postgresql://...
    python -m experimental.grid_search_lab HDB
    python -m experimental.grid_search_lab HDB --von 1000000000 --top 15
"""

from __future__ import annotations

import argparse
import itertools
import sys

import pandas as pd

from experimental.indicator_lab import (
    BO_LOC_MAC_DINH,
    TRAILING_TP_TIERS_MAC_DINH,
    InvalidIndicatorLabError,
    chay_backtest,
    chay_backtest_ket_hop_nhieu_bo,
)
from main import load_config, resolve_storage_path
from core.storage import Storage

# ==============================================================================
# LƯỚI THAM SỐ THỬ NGHIỆM — chỉnh trực tiếp ở đây nếu muốn mở rộng/thu
# hẹp phạm vi dò. Giữ NGUYÊN công thức tín hiệu gốc — chỉ đổi GIÁ TRỊ.
# ==============================================================================

LUOI_THAM_SO = {
    "buy_lookback": [3, 4, 6, 8],
    "range_pct_max": [3.0, 5.0, 7.0, 10.0],
    "body_pct_min": [0.5, 1.0],
    "body_pct_max": [4.0, 6.0, 8.0, 10.0],
    "ema_period": [100, 150, 200],
    "ma_period": [10, 20, 30],
}


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


def chay_grid_search(
    df_ohlcv: pd.DataFrame,
    von_ban_dau: float = 1_000_000_000,
    ty_trong_von_pct: float = 50.0,
    sell_lookback_bang_buy_lookback: bool = True,
    chuoi_giai_doan=None,
    giai_doan_loc: str = None,
    hien_tien_do: bool = True,
) -> list[dict]:
    """Thử TOÀN BỘ tổ hợp trong `LUOI_THAM_SO` — engine chỉ giao dịch
    LONG theo đúng thực tế TTCK VN (không bán khống), trả về danh sách
    kết quả đã sắp theo lợi nhuận ròng giảm dần.

    `chuoi_giai_doan` + `giai_doan_loc` (bổ sung 06/08/2026): nếu truyền
    cả 2, CHỈ tính các lần vào lệnh khớp đúng giai đoạn thị trường/ngành
    mong muốn (xem `core.market_regime_detector.tinh_chuoi_giai_doan_theo_ngay`
    và `chay_backtest`).
    """
    ket_qua: list[dict] = []
    cac_khoa = list(LUOI_THAM_SO.keys())
    cac_gia_tri = list(LUOI_THAM_SO.values())
    tong_to_hop = 1
    for gt in cac_gia_tri:
        tong_to_hop *= len(gt)

    if hien_tien_do:
        print(f"Tổng số tổ hợp cần thử: {tong_to_hop}")
    da_thu = 0

    for combo in itertools.product(*cac_gia_tri):
        da_thu += 1
        tham_so = dict(zip(cac_khoa, combo))
        if tham_so["body_pct_min"] >= tham_so["body_pct_max"]:
            continue  # tổ hợp vô lý, bỏ qua
        tham_so["sell_lookback"] = (
            tham_so["buy_lookback"] if sell_lookback_bang_buy_lookback else tham_so["buy_lookback"]
        )

        try:
            kq = chay_backtest(
                df_ohlcv, tham_so, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
                von_ban_dau=von_ban_dau, ty_trong_von_pct=ty_trong_von_pct,
                chuoi_giai_doan=chuoi_giai_doan, giai_doan_loc=giai_doan_loc,
            )
        except InvalidIndicatorLabError:
            continue

        if kq["so_lenh_da_dong"] == 0:
            continue  # không có lệnh LONG nào -> bỏ qua, không xếp hạng

        ket_qua.append({
            "buy_lookback": tham_so["buy_lookback"],
            "range_pct_max": tham_so["range_pct_max"],
            "body_pct_min": tham_so["body_pct_min"],
            "body_pct_max": tham_so["body_pct_max"],
            "ema_period": tham_so["ema_period"],
            "ma_period": tham_so["ma_period"],
            "so_lenh_da_dong": kq["so_lenh_da_dong"],
            "win_rate_pct": kq["win_rate_pct"],
            "loi_nhuan_rong": kq["loi_nhuan_rong"],
            "loi_nhuan_rong_pct": kq["loi_nhuan_rong_pct"],
            "co_vi_the_dang_mo": kq["open_position"] is not None,
        })

        if hien_tien_do and da_thu % 200 == 0:
            print(f"  ... đã thử {da_thu}/{tong_to_hop} tổ hợp")

    ket_qua.sort(key=lambda r: r["loi_nhuan_rong"], reverse=True)
    return ket_qua


def main() -> None:
    parser = argparse.ArgumentParser(description="Dò tham số Phòng thí nghiệm chỉ báo (chỉ tính LONG)")
    parser.add_argument("symbol", help="Mã cổ phiếu, VD: HDB")
    parser.add_argument("--von", type=float, default=1_000_000_000, help="Vốn ban đầu (VNĐ)")
    parser.add_argument("--ty-trong", type=float, default=50.0, help="%% vốn dùng mỗi lệnh")
    parser.add_argument("--top", type=int, default=10, help="Số dòng kết quả tốt nhất hiển thị")
    parser.add_argument(
        "--min-lenh", type=int, default=0,
        help="CHỈ xếp hạng các bộ tham số có SỐ LỆNH >= giá trị này — dùng để tìm bộ "
             "tham số cho NHIỀU LỆNH HƠN mà vẫn tối ưu lợi nhuận (VD --min-lenh 25).",
    )
    parser.add_argument(
        "--ket-hop-top", type=int, default=0,
        help="Sau khi xếp hạng, TỰ ĐỘNG chạy thêm 1 backtest KẾT HỢP (logic OR) đúng "
             "N bộ tham số đứng đầu (VD --ket-hop-top 10 để kết hợp Top 10) — trả lời "
             "câu hỏi 'vào lệnh nếu đạt tiêu chí BẤT KỲ bộ nào trong N bộ tốt nhất'. "
             "Để 0 (mặc định) để bỏ qua bước này.",
    )
    args = parser.parse_args()

    config = load_config()
    storage = Storage(db_path=resolve_storage_path(config))
    df = _load_ohlcv(storage, args.symbol.upper())
    storage.close()

    print(f"Đã tải {len(df)} phiên OHLCV cho {args.symbol.upper()} "
          f"({df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()})")
    print(
        "\n⚠️ CẢNH BÁO OVERFIT: kết quả dưới đây dò trên CÙNG 1 bộ dữ liệu lịch sử — "
        "bộ tham số xếp hạng cao KHÔNG đảm bảo lặp lại hiệu suất trong tương lai. "
        "Dùng để THAM KHẢO thu hẹp phạm vi, không phải kết luận cuối cùng.\n"
    )

    ket_qua = chay_grid_search(df, von_ban_dau=args.von, ty_trong_von_pct=args.ty_trong)

    if not ket_qua:
        print("Không tìm thấy tổ hợp nào có ít nhất 1 lệnh LONG trong dữ liệu.")
        return

    if args.min_lenh > 0:
        so_luong_truoc_loc = len(ket_qua)
        ket_qua = [r for r in ket_qua if r["so_lenh_da_dong"] >= args.min_lenh]
        print(f"Đã lọc: chỉ giữ các bộ có >= {args.min_lenh} lệnh ({len(ket_qua)}/{so_luong_truoc_loc} bộ còn lại).")
        if not ket_qua:
            print(f"Không có bộ tham số nào đạt >= {args.min_lenh} lệnh. Thử giảm --min-lenh.")
            return

    df_ket_qua = pd.DataFrame(ket_qua)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(f"\n=== TOP {args.top} BỘ THAM SỐ CHO LONG (theo lợi nhuận ròng, vốn {args.von:,.0f}đ) ===\n")
    print(df_ket_qua.head(args.top).to_string(index=False))

    top1 = ket_qua[0]
    print(f"\n=== BỘ THAM SỐ TỐT NHẤT ===")
    print(f"buy_lookback = sell_lookback = {top1['buy_lookback']}")
    print(f"range_pct_max = {top1['range_pct_max']}")
    print(f"body_pct_min = {top1['body_pct_min']}, body_pct_max = {top1['body_pct_max']}")
    print(f"ema_period = {top1['ema_period']}, ma_period = {top1['ma_period']}")
    print(f"-> {top1['so_lenh_da_dong']} lệnh LONG, win rate {top1['win_rate_pct']}%, "
          f"lợi nhuận ròng {top1['loi_nhuan_rong']:,.0f}đ ({top1['loi_nhuan_rong_pct']:+.2f}%)")
    if top1["co_vi_the_dang_mo"]:
        print("(Lưu ý: bộ này còn 1 vị thế LONG đang mở tại cuối dữ liệu, chưa tính vào lợi nhuận ròng trên.)")

    if args.ket_hop_top > 0:
        so_bo_thuc_te = min(args.ket_hop_top, len(ket_qua))
        danh_sach_tham_so_ket_hop = [
            {
                "buy_lookback": r["buy_lookback"], "sell_lookback": r["buy_lookback"],
                "ema_period": r["ema_period"], "ma_period": r["ma_period"],
                "range_pct_max": r["range_pct_max"],
                "body_pct_min": r["body_pct_min"], "body_pct_max": r["body_pct_max"],
            }
            for r in ket_qua[:so_bo_thuc_te]
        ]
        print(f"\n=== KẾT HỢP TOP {so_bo_thuc_te} BỘ THAM SỐ (logic OR — vào lệnh nếu đạt BẤT KỲ bộ nào) ===")
        try:
            kq_ket_hop = chay_backtest_ket_hop_nhieu_bo(
                df, danh_sach_tham_so_ket_hop, BO_LOC_MAC_DINH, TRAILING_TP_TIERS_MAC_DINH,
                von_ban_dau=args.von, ty_trong_von_pct=args.ty_trong,
            )
        except InvalidIndicatorLabError as exc:
            print(f"Lỗi khi chạy kết hợp: {exc}")
            return

        print(f"-> {kq_ket_hop['so_lenh_da_dong']} lệnh LONG, win rate {kq_ket_hop['win_rate_pct']}%, "
              f"lợi nhuận ròng {kq_ket_hop['loi_nhuan_rong']:,.0f}đ ({kq_ket_hop['loi_nhuan_rong_pct']:+.2f}%)")
        if kq_ket_hop["open_position"]:
            print("(Lưu ý: còn 1 vị thế LONG đang mở tại cuối dữ liệu, chưa tính vào lợi nhuận ròng trên.)")

        print(f"\nSo sánh với bộ tốt nhất ĐƠN LẺ (#1): {top1['so_lenh_da_dong']} lệnh, "
              f"lợi nhuận ròng {top1['loi_nhuan_rong']:,.0f}đ ({top1['loi_nhuan_rong_pct']:+.2f}%)")
        chenh_lech_lenh = kq_ket_hop["so_lenh_da_dong"] - top1["so_lenh_da_dong"]
        chenh_lech_loi_nhuan = kq_ket_hop["loi_nhuan_rong"] - top1["loi_nhuan_rong"]
        print(f"Chênh lệch: {chenh_lech_lenh:+d} lệnh, {chenh_lech_loi_nhuan:+,.0f}đ lợi nhuận ròng")


if __name__ == "__main__":
    main()
