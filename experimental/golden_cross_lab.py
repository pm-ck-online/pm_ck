"""
experimental/golden_cross_lab.py
====================================
Script backtest chiến lược Golden Cross (MA20 cắt lên MA50) / Death
Cross (MA20 cắt xuống MA100) — kèm Stop Loss an toàn — trên TOÀN BỘ
lịch sử của 1 mã, in ra ĐẦY ĐỦ danh sách các lần giao dịch để kiểm
chứng (không chỉ 1 ví dụ đơn lẻ).

⚠️ CẢNH BÁO: kết quả CHỈ mang tính THAM KHẢO trên dữ liệu lịch sử,
KHÔNG đảm bảo lặp lại hiệu suất tương lai.

CÁCH CHẠY:
    set PM_CK_DB_PATH=postgresql://...
    python -m experimental.golden_cross_lab HDB
    python -m experimental.golden_cross_lab HDB --ma-nhanh 20 --ma-cham 50 --ma-dai-han 100 --stop-loss 10
"""

from __future__ import annotations

import argparse

import pandas as pd

from experimental.indicator_lab import InvalidIndicatorLabError, chay_backtest_ma_crossover
from main import load_config, resolve_storage_path
from core.storage import Storage


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
    parser = argparse.ArgumentParser(description="Backtest chiến lược Golden Cross / Death Cross")
    parser.add_argument("symbol", help="Mã cổ phiếu, VD: HDB")
    parser.add_argument("--ma-nhanh", type=int, default=20, help="Chu kỳ MA nhanh (mặc định 20)")
    parser.add_argument("--ma-cham", type=int, default=50, help="Chu kỳ MA chậm — vào lệnh khi MA nhanh cắt lên (mặc định 50)")
    parser.add_argument("--ma-dai-han", type=int, default=100, help="Chu kỳ MA dài hạn — thoát lệnh khi MA nhanh cắt xuống (mặc định 100)")
    parser.add_argument("--stop-loss", type=float, default=10.0, help="%% lỗ tối đa an toàn (mặc định 10%%)")
    parser.add_argument("--von", type=float, default=1_000_000_000, help="Vốn ban đầu (VNĐ)")
    parser.add_argument("--ty-trong", type=float, default=50.0, help="%% vốn dùng mỗi lệnh")
    args = parser.parse_args()

    config = load_config()
    storage = Storage(db_path=resolve_storage_path(config))
    df = _load_ohlcv(storage, args.symbol.upper())
    storage.close()

    print(f"Đã tải {len(df)} phiên OHLCV cho {args.symbol.upper()} "
          f"({df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()})")
    print(f"Chiến lược: MA{args.ma_nhanh} cắt lên MA{args.ma_cham} = VÀO LỆNH | "
          f"MA{args.ma_nhanh} cắt xuống MA{args.ma_dai_han} HOẶC lỗ {args.stop_loss}% = THOÁT LỆNH")
    print(
        "\n⚠️ CẢNH BÁO: kết quả dưới đây CHỈ mang tính THAM KHẢO trên dữ liệu lịch sử — "
        "KHÔNG đảm bảo lặp lại hiệu suất tương lai.\n"
    )

    try:
        kq = chay_backtest_ma_crossover(
            df, ma_nhanh_period=args.ma_nhanh, ma_cham_period=args.ma_cham,
            ma_dai_han_period=args.ma_dai_han, stop_loss_pct=args.stop_loss,
            von_ban_dau=args.von, ty_trong_von_pct=args.ty_trong,
        )
    except InvalidIndicatorLabError as exc:
        raise SystemExit(f"Lỗi tham số: {exc}")

    if kq["so_lenh_da_dong"] == 0 and not kq["open_position"]:
        print("Không tìm thấy lần Golden Cross nào trong lịch sử đã quét.")
        return

    print(f"=== THỐNG KÊ TỔNG THỂ ===")
    print(f"Tổng số lệnh đã đóng: {kq['so_lenh_da_dong']}")
    print(f"Thắng/Thua: {kq['so_lan_thang']}/{kq['so_lan_thua']} (Win rate: {kq['win_rate_pct']}%)")
    print(f"Thoát bởi Stop Loss: {kq['so_lan_thoat_boi_stop_loss']} | Thoát bởi Death Cross: {kq['so_lan_thoat_boi_death_cross']}")
    print(f"Vốn cuối cùng: {kq['von_cuoi_cung']:,.0f}đ")
    print(f"Lợi nhuận ròng: {kq['loi_nhuan_rong']:,.0f}đ ({kq['loi_nhuan_rong_pct']:+.2f}%)")

    if kq["trades"]:
        print(f"\n=== TOÀN BỘ {len(kq['trades'])} LẦN GIAO DỊCH (để bạn tự đối chiếu) ===")
        df_trades = pd.DataFrame(kq["trades"])[
            ["entry_date", "entry_price", "exit_date", "exit_price", "final_pnl_pct", "ly_do_thoat"]
        ]
        pd.set_option("display.width", 200)
        pd.set_option("display.max_rows", None)
        print(df_trades.to_string(index=False))

    if kq["open_position"]:
        op = kq["open_position"]
        print(f"\n=== VỊ THẾ ĐANG MỞ (chưa tính vào lợi nhuận ròng ở trên) ===")
        print(f"Vào ngày {op['entry_date']} tại giá {op['entry_price']}, hiện tại "
              f"({op['as_of_date']}) giá {op['current_price']} — PnL tạm tính {op['unrealized_pnl_pct']:+.2f}%")


if __name__ == "__main__":
    main()
