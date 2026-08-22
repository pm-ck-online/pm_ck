"""
experimental/tong_hop_toi_uu_theo_giai_doan.py
==================================================
Script TỔNG HỢP — tự động dò tham số tối ưu (LONG, lợi nhuận ròng cao
nhất trên vốn ban đầu) cho NHIỀU MÃ, tách riêng theo TỪNG GIAI ĐOẠN
ngành (Uptrend/Sideway/Downtrend) — trả lời câu hỏi "mã này nên dùng bộ
tham số nào khi ngành đang ở giai đoạn nào".

Kết hợp 2 công cụ đã có:
  - `experimental/grid_search_lab.py` (dò tham số LONG tối ưu)
  - `core/market_regime_detector.tinh_chuoi_giai_doan_theo_ngay()` (phát
    hiện giai đoạn ngành theo từng ngày)

⚠️ CẢNH BÁO OVERFIT + THỜI GIAN CHẠY: mỗi (mã × giai đoạn) là 1 lượt dò
tham số đầy đủ — với 3 mã × 3 giai đoạn = 9 lượt, có thể mất khá lâu
(vài chục phút tùy độ lớn LƯỚI_THAM_SO_NHANH bên dưới — ĐÃ THU GỌN sẵn
so với `grid_search_lab.py` gốc để giữ tổng thời gian chạy hợp lý). Kết
quả CHỈ mang tính THAM KHẢO, KHÔNG đảm bảo lặp lại hiệu suất tương lai.

CÁCH CHẠY:
    set PM_CK_DB_PATH=postgresql://...
    python -m experimental.tong_hop_toi_uu_theo_giai_doan
    python -m experimental.tong_hop_toi_uu_theo_giai_doan --ma HDB SSI CEO --von 1000000000
"""

from __future__ import annotations

import argparse

import pandas as pd

from experimental.grid_search_lab import LUOI_THAM_SO, chay_grid_search
from core.market_regime_detector import tinh_chuoi_giai_doan_theo_ngay
from main import load_config, resolve_storage_path
from core.storage import Storage

# Lưới tham số THU GỌN (so với grid_search_lab.py gốc ~1.152 tổ hợp) —
# vì chạy tới 9 lần (3 mã x 3 giai đoạn), cần rút ngắn lưới để tổng thời
# gian chạy hợp lý. Có thể tự chỉnh lại nếu muốn dò kỹ hơn (đổi trực
# tiếp các list bên dưới).
LUOI_THAM_SO_NHANH = {
    "buy_lookback": [3, 4, 6],
    "range_pct_max": [5.0, 10.0],
    "body_pct_min": [0.5, 1.0],
    "body_pct_max": [6.0, 10.0],
    "ema_period": [100, 200],
    "ma_period": [10, 20],
}

GIAI_DOAN_CAN_DO = ["uptrend", "sideway", "downtrend"]


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


def _lay_chuoi_giai_doan_theo_nganh(storage: Storage, symbol: str) -> pd.Series:
    """Ưu tiên đọc bản ĐÃ LƯU SẴN (tính qua run_full_market.py hằng
    ngày) — nếu chưa có, tính lại trực tiếp (chậm hơn)."""
    sector_record = storage.get_latest("symbol_sector", symbol)
    nganh = sector_record["data"].get("sector") if sector_record else None
    if nganh is None:
        raise SystemExit(f"Không xác định được ngành của mã '{symbol}'.")

    da_luu = storage.get_latest("chuoi_giai_doan_lich_su", nganh)
    if da_luu is not None:
        records = da_luu["data"].get("records", [])
        if records:
            df_tam = pd.DataFrame(records)
            df_tam["date"] = pd.to_datetime(df_tam["date"])
            return pd.Series(df_tam["giai_doan"].values, index=df_tam["date"])

    print(f"  (chưa có bản lưu sẵn giai đoạn ngành '{nganh}' — tính lại trực tiếp...)")
    all_keys = storage.query_all_keys("symbol_sector")
    sector_map = storage.get_latest_many("symbol_sector", all_keys)
    ma_cung_nganh = [ma for ma, rec in sector_map.items() if rec["data"].get("sector") == nganh]
    ohlcv_map = storage.get_latest_many("ohlcv_history", ma_cung_nganh)
    du_lieu_theo_ma = {}
    for ma, rec in ohlcv_map.items():
        recs = rec["data"].get("records", [])
        if recs:
            df_ma = pd.DataFrame(recs)
            df_ma["date"] = pd.to_datetime(df_ma["date"])
            du_lieu_theo_ma[ma] = df_ma.sort_values("date").reset_index(drop=True)
    return tinh_chuoi_giai_doan_theo_ngay(du_lieu_theo_ma)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tổng hợp tối ưu tham số LONG theo giai đoạn ngành, nhiều mã")
    parser.add_argument("--ma", nargs="+", default=["HDB", "SSI", "CEO"], help="Danh sách mã (mặc định HDB SSI CEO)")
    parser.add_argument("--von", type=float, default=1_000_000_000, help="Vốn ban đầu (VNĐ)")
    parser.add_argument("--ty-trong", type=float, default=50.0, help="%% vốn dùng mỗi lệnh")
    args = parser.parse_args()

    global LUOI_THAM_SO
    import experimental.grid_search_lab as gsl
    gsl.LUOI_THAM_SO = LUOI_THAM_SO_NHANH

    config = load_config()
    storage = Storage(db_path=resolve_storage_path(config))

    print(
        "\n⚠️ CẢNH BÁO OVERFIT: đây là kết quả dò trên dữ liệu lịch sử, KHÔNG đảm bảo "
        "lặp lại hiệu suất tương lai. Dùng để THAM KHẢO, không phải kết luận cuối cùng.\n"
    )

    ket_qua_tong_hop: dict[str, dict[str, dict]] = {}

    for ma in args.ma:
        ma = ma.upper()
        print(f"\n{'=' * 70}\n=== MÃ {ma} ===\n{'=' * 70}")
        df = _load_ohlcv(storage, ma)
        print(f"Đã tải {len(df)} phiên OHLCV ({df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()})")

        chuoi_giai_doan = _lay_chuoi_giai_doan_theo_nganh(storage, ma)
        print(f"Chuỗi giai đoạn ngành: {len(chuoi_giai_doan)} ngày có dữ liệu — "
              f"phân bố: {dict(chuoi_giai_doan.value_counts())}")

        ket_qua_tong_hop[ma] = {}
        for giai_doan in GIAI_DOAN_CAN_DO:
            print(f"\n--- {ma} | Giai đoạn: {giai_doan.upper()} ---")
            ket_qua = chay_grid_search(
                df, von_ban_dau=args.von, ty_trong_von_pct=args.ty_trong,
                chuoi_giai_doan=chuoi_giai_doan, giai_doan_loc=giai_doan,
                hien_tien_do=False,
            )
            if not ket_qua:
                print(f"  Không tìm thấy bộ tham số nào có lệnh LONG trong giai đoạn {giai_doan}.")
                ket_qua_tong_hop[ma][giai_doan] = None
                continue
            top1 = ket_qua[0]
            print(f"  Tốt nhất: buy_lookback={top1['buy_lookback']}, ema={top1['ema_period']}, "
                  f"ma={top1['ma_period']}, range_max={top1['range_pct_max']}, "
                  f"body_max={top1['body_pct_max']} -> {top1['so_lenh_da_dong']} lệnh, "
                  f"win rate {top1['win_rate_pct']}%, lợi nhuận ròng "
                  f"{top1['loi_nhuan_rong']:,.0f}đ ({top1['loi_nhuan_rong_pct']:+.2f}%)")
            ket_qua_tong_hop[ma][giai_doan] = top1

    storage.close()

    # --- Bảng tổng kết cuối cùng ---
    print(f"\n\n{'=' * 90}\n=== BẢNG TỔNG KẾT — BỘ THAM SỐ TỐI ƯU LONG THEO GIAI ĐOẠN NGÀNH (vốn {args.von:,.0f}đ) ===\n{'=' * 90}")
    rows = []
    for ma, theo_giai_doan in ket_qua_tong_hop.items():
        for giai_doan in GIAI_DOAN_CAN_DO:
            r = theo_giai_doan.get(giai_doan)
            if r is None:
                rows.append({"Mã": ma, "Giai đoạn": giai_doan, "Số lệnh": 0, "Win rate": None,
                             "Lợi nhuận ròng": None, "Lợi nhuận %": None, "Tham số": "Không có tín hiệu"})
            else:
                tham_so_txt = (f"buy_lb={r['buy_lookback']}, ema={r['ema_period']}, ma={r['ma_period']}, "
                                f"range_max={r['range_pct_max']}, body_max={r['body_pct_max']}")
                rows.append({
                    "Mã": ma, "Giai đoạn": giai_doan, "Số lệnh": r["so_lenh_da_dong"],
                    "Win rate": r["win_rate_pct"], "Lợi nhuận ròng": r["loi_nhuan_rong"],
                    "Lợi nhuận %": r["loi_nhuan_rong_pct"], "Tham số": tham_so_txt,
                })

    df_tong_ket = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(df_tong_ket.to_string(index=False))


if __name__ == "__main__":
    main()
