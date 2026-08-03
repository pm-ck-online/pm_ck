"""
update_vcp.py
===============
Tính rà soát mô hình co hẹp biên độ (VCP) cho XAUUSD và BTC/USD, LƯU kết
quả vào Supabase (category "vcp_scan_result", key = symbol) để dashboard
đọc và hiển thị.

KHÔNG cần API key Binance. CẦN đặt PM_CK_DB_PATH trỏ tới Supabase.

CÁCH CHẠY (giống các script trước):
    set PM_CK_DB_PATH=postgresql://...
    python update_vcp.py
"""

from __future__ import annotations

import logging

from main import load_config, resolve_storage_path
from core.storage import Storage
from core.data_collector import BinanceDataSource
from core.volatility_contraction_scanner import rao_soat_mo_hinh_co_hep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pm_ck.update_vcp")

# BTC biến động mạnh hơn XAUUSD rất nhiều -> dùng bộ ngưỡng phân bậc biên
# độ RIÊNG cho từng symbol (mục 10.3 tài liệu gốc).
NGUONG_BAC_THEO_SYMBOL = {
    "XAUUSD": [20.0, 15.0, 10.0, 5.0, 3.0],
    "BTCUSD": [40.0, 30.0, 20.0, 10.0, 5.0],
}


def main() -> None:
    print("pm_ck — Rà soát mô hình co hẹp biên độ (VCP) cho XAUUSD/BTCUSD")

    config = load_config()
    storage = Storage(db_path=resolve_storage_path(config))
    source = BinanceDataSource()

    for symbol in ("XAUUSD", "BTCUSD"):
        logger.info("=== Đang rà soát %s ===", symbol)
        try:
            result = rao_soat_mo_hinh_co_hep(
                symbol,
                source.fetch_ohlcv,
                khung_thoi_gian_ung_vien=("1d", "4h"),
                so_ngay_tham_chieu=45,
                so_chu_ky_toi_thieu=3,
                nguong_bac_bien_do=NGUONG_BAC_THEO_SYMBOL.get(symbol),
            )
            storage.save("vcp_scan_result", symbol, result)
            logger.info(
                "%s: xác nhận co hẹp = %s, khung = %s",
                symbol, result["xac_nhan_co_hep"], result["khung_thoi_gian_da_chon"],
            )
        except Exception:
            logger.exception("Lỗi khi rà soát %s — bỏ qua, tiếp tục symbol tiếp theo.", symbol)
            continue

    storage.close()
    print("\nHoàn tất. Vào dashboard -> mục 'Rà soát mô hình co hẹp (XAUUSD/BTC)' để xem kết quả.")


if __name__ == "__main__":
    main()
