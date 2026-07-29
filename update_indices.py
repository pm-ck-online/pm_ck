"""
update_indices.py
=====================
Script NHỎ, chỉ cập nhật dữ liệu CHỈ SỐ THỊ TRƯỜNG CHUNG (VNINDEX, VN30,
VN100...) — tách riêng khỏi `main.py` để KHÔNG chạy lại toàn bộ danh sách
cổ phiếu trong watchlist.symbols (tránh bị rate limit lần 2 vì main.py
không có độ trễ giữa các lần gọi).

CÁCH CHẠY (sau khi đã chạy xong run_full_market.py cho phần cổ phiếu):
    python update_indices.py
"""

from __future__ import annotations

import logging

from main import build_data_collector, load_config, resolve_storage_path, run_index_step
from core.storage import Storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pm_ck.update_indices")


def main() -> None:
    print("pm_ck — Cập nhật dữ liệu chỉ số thị trường chung (VNINDEX/VN30/VN100)")

    config = load_config()
    storage = Storage(db_path=resolve_storage_path(config))
    collector = build_data_collector(config)

    index_symbols = config.get("watchlist", {}).get("indices", [])
    if not index_symbols:
        logger.warning("Chưa cấu hình watchlist.indices trong config.yaml.")
        return

    for index_symbol in index_symbols:
        logger.info("=== Xử lý chỉ số %s ===", index_symbol)
        try:
            run_index_step(collector, storage, index_symbol, config)
        except Exception:
            logger.exception(
                "Lỗi khi xử lý chỉ số %s — bỏ qua, tiếp tục chỉ số tiếp theo.",
                index_symbol,
            )
            continue

    storage.close()
    logger.info("Hoàn tất cập nhật chỉ số thị trường chung.")


if __name__ == "__main__":
    main()
