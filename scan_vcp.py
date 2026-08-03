"""
scan_vcp.py
=============
Script chạy thử module rà soát mô hình co hẹp biên độ (VCP) cho XAUUSD
(vàng thế giới, qua PAXGUSDT trên Binance) và BTC/USD (qua BTCUSDT).

CÁCH CHẠY:
    python scan_vcp.py

KHÔNG cần PM_CK_DB_PATH / kết nối Supabase — script này CHỈ in kết quả ra
màn hình, không ghi vào database. Cũng KHÔNG cần API key (Binance Public
API cho endpoint `klines` là dữ liệu công khai, miễn phí).
"""

from __future__ import annotations

import json

from core.data_collector import BinanceDataSource
from core.volatility_contraction_scanner import rao_soat_mo_hinh_co_hep

# BTC biến động mạnh hơn XAUUSD rất nhiều -> dùng bộ ngưỡng phân bậc biên
# độ RIÊNG cho từng symbol (mục 10.3 tài liệu gốc).
NGUONG_BAC_THEO_SYMBOL = {
    "XAUUSD": [20.0, 15.0, 10.0, 5.0, 3.0],   # mặc định, phù hợp vàng
    "BTCUSD": [40.0, 30.0, 20.0, 10.0, 5.0],  # BTC biến động mạnh hơn nhiều
}


def main() -> None:
    source = BinanceDataSource()

    for symbol in ("XAUUSD", "BTCUSD"):
        print(f"\n{'=' * 70}\nRà soát mô hình co hẹp — {symbol}\n{'=' * 70}")
        try:
            result = rao_soat_mo_hinh_co_hep(
                symbol,
                source.fetch_ohlcv,
                khung_thoi_gian_ung_vien=("1d", "4h"),
                so_ngay_tham_chieu=45,
                so_chu_ky_toi_thieu=3,
                nguong_bac_bien_do=NGUONG_BAC_THEO_SYMBOL.get(symbol),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Lỗi khi rà soát {symbol}: {exc}")
            continue

        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
