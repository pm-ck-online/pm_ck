"""
run_daily_breakout_screener.py
================================
Chạy `core/daily_breakout_backtest.py` cho TOÀN BỘ watchlist, đối chiếu
CẢ 2 cấu hình (CAU_HINH_GOC giữ nguyên số nến H1, CAU_HINH_SCALE_THOI_GIAN
scale theo tầm nhìn thời gian tương đương) — đúng yêu cầu gốc: KHÔNG giả
định trước cấu hình nào tốt hơn, phải backtest và đối chiếu số liệu thật.

Dùng lại `ohlcv_history` ĐÃ CÓ SẴN trong storage — KHÔNG gọi thêm API.

CÁCH CHẠY:
    python run_daily_breakout_screener.py
    python run_daily_breakout_screener.py --khong-loc-thanh-khoan
    python run_daily_breakout_screener.py --von-ban-dau 500000000
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import yaml

from core.daily_breakout_backtest import (
    CAU_HINH_GOC,
    CAU_HINH_SCALE_THOI_GIAN,
    backtest_chien_luoc_breakout,
    tong_hop_ket_qua,
)
from core.storage import Storage
from main import load_config, resolve_storage_path

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pm_ck.run_daily_breakout_screener")

SO_PHIEN_TOI_THIEU = 220  # cần dư ~ CAU_HINH_GOC["ema_period"]=150 để EMA150 có ý nghĩa


def _chay_1_cau_hinh(
    danh_sach_ma: list[str], storage: Storage, cau_hinh: dict, ten_cau_hinh: str,
    initial_capital: float, yeu_cau_thanh_khoan: bool, gia_tri_trung_binh_toi_thieu: float,
) -> pd.DataFrame:
    hang = []
    for ma in danh_sach_ma:
        record = storage.get_latest("ohlcv_history", ma)
        if record is None:
            continue
        records = record["data"].get("records", [])
        if len(records) < SO_PHIEN_TOI_THIEU:
            continue
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        try:
            lenh_list = backtest_chien_luoc_breakout(
                df, cau_hinh=cau_hinh, yeu_cau_thanh_khoan=yeu_cau_thanh_khoan,
                gia_tri_trung_binh_toi_thieu=gia_tri_trung_binh_toi_thieu,
            )
        except Exception:
            logger.exception("[%s] Lỗi khi backtest (%s) — bỏ qua.", ma, ten_cau_hinh)
            continue
        tong_hop = tong_hop_ket_qua(lenh_list, initial_capital=initial_capital)
        hang.append({"Mã": ma, **tong_hop})

    return pd.DataFrame(hang)


def main() -> None:
    parser = argparse.ArgumentParser(description="Screener chiến lược breakout nến ngày, đối chiếu 2 cấu hình.")
    parser.add_argument("--von-ban-dau", type=float, default=1_000_000_000.0)
    parser.add_argument("--khong-loc-thanh-khoan", action="store_true")
    parser.add_argument("--gia-tri-thanh-khoan-toi-thieu", type=float, default=1_000_000_000.0)
    args = parser.parse_args()

    config = load_config()
    storage = Storage(db_path=resolve_storage_path(config))

    with open("config/config.yaml", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)
    danh_sach_ma = list(cfg_raw.get("watchlist", {}).get("symbols", {}).keys())
    print(f"Tổng số mã trong watchlist: {len(danh_sach_ma)}")

    yeu_cau_thanh_khoan = not args.khong_loc_thanh_khoan
    for ten_cau_hinh, cau_hinh in [
        ("CAU_HINH_GOC (EMA150/MA20/lookback2, giữ nguyên số nến H1)", CAU_HINH_GOC),
        ("CAU_HINH_SCALE_THOI_GIAN (EMA38/MA5/lookback2)", CAU_HINH_SCALE_THOI_GIAN),
    ]:
        print(f"\n=== {ten_cau_hinh} ===")
        df_ket_qua = _chay_1_cau_hinh(
            danh_sach_ma, storage, cau_hinh, ten_cau_hinh,
            args.von_ban_dau, yeu_cau_thanh_khoan, args.gia_tri_thanh_khoan_toi_thieu,
        )
        if df_ket_qua.empty:
            print("Không có mã nào đủ dữ liệu.")
            continue

        co_lenh = df_ket_qua[df_ket_qua["n_trades"] > 0].copy()
        print(f"Số mã đủ dữ liệu: {len(df_ket_qua)}, số mã có ít nhất 1 lệnh: {len(co_lenh)}")
        if not co_lenh.empty:
            print(f"Tổng số lệnh: {int(co_lenh['n_trades'].sum())}")
            print(f"Win rate trung bình (không trọng số): {co_lenh['win_rate_pct'].mean():.2f}%")
            print(f"Lãi cộng dồn trung bình/mã: {co_lenh['total_return_pct'].mean():.2f}%")
            top10 = co_lenh.sort_values("total_return_pct", ascending=False).head(10)
            print("Top 10 mã theo lãi cộng dồn:")
            print(top10[["Mã", "n_trades", "win_rate_pct", "total_return_pct", "profit_factor", "max_drawdown_pct"]]
                  .to_string(index=False))

        out_path = f"daily_breakout_{'goc' if cau_hinh is CAU_HINH_GOC else 'scale_thoi_gian'}.csv"
        df_ket_qua.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Đã lưu chi tiết: {out_path}")

    storage.close()


if __name__ == "__main__":
    main()
