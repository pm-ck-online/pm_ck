"""
fix_symbol_sector.py
=====================
Dọn dẹp DỨT ĐIỂM dữ liệu ngành (symbol_sector) trong Supabase — khắc phục
tình trạng dashboard hiện các ngành tiếng Việt lạ (VD: "Bán buôn", "Bán
lẻ"...) xen lẫn với 17 ngành tiếng Anh chuẩn hóa trong config.yaml.

NGUYÊN NHÂN: bảng `symbol_sector` trong Supabase là kiểu "lưu mãi mãi,
không tự xóa" (append-only). Ở một thời điểm trước đây, hệ thống từng
lấy TOÀN BỘ ~1.500+ mã trên thị trường kèm phân loại ngành TIẾNG VIỆT
gốc của vnstock (khi config.yaml chưa giới hạn xuống đúng 212 mã theo
dõi). Từ đó, `run_full_market.py` chỉ xử lý lại đúng danh sách mã hiện
có trong config.yaml -> các mã NGOÀI danh sách đó (không còn theo dõi)
vẫn giữ nguyên nhãn ngành tiếng Việt cũ trong Supabase, khiến dashboard
liệt kê chúng như những "ngành" riêng biệt, dù thực chất không liên quan
gì tới 212 mã đang theo dõi.

SCRIPT NÀY LÀM 2 VIỆC (chạy trong vài giây, KHÔNG gọi API, KHÔNG rate limit):
    1. Với đúng 212 mã trong config.yaml -> ghi đè lại "symbol_sector"
       bằng ĐÚNG giá trị tiếng Anh chuẩn hóa (đảm bảo dữ liệu luôn khớp,
       kể cả khi trước đó bị ghi nhầm).
    2. Với mọi mã KHÁC (không còn nằm trong config.yaml) -> XÓA hẳn bản
       ghi "symbol_sector" của mã đó — loại bỏ toàn bộ ngành lạ/cũ.

CÁCH CHẠY (giống các script trước, cần đặt PM_CK_DB_PATH trỏ tới Supabase):
    set PM_CK_DB_PATH=postgresql://...
    python fix_symbol_sector.py
"""

from __future__ import annotations

import logging

from main import load_config, resolve_storage_path
from core.storage import Storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pm_ck.fix_symbol_sector")


def main() -> None:
    print("pm_ck — Dọn dẹp dữ liệu ngành (symbol_sector) trong Supabase")

    config = load_config()
    storage = Storage(db_path=resolve_storage_path(config))

    watchlist_symbols: dict[str, str] = config.get("watchlist", {}).get("symbols", {})
    if not watchlist_symbols:
        logger.error("config.yaml không có watchlist.symbols — dừng lại để tránh xóa nhầm toàn bộ.")
        return

    logger.info("Danh sách hiện tại trong config.yaml: %d mã.", len(watchlist_symbols))

    # --- Bước 1: ghi đè lại ĐÚNG 212 mã đang theo dõi bằng sector chuẩn ---
    db_path = resolve_storage_path(config)
    so_da_ghi_de = 0
    for symbol, sector in watchlist_symbols.items():
        for attempt in (1, 2):
            try:
                storage.save("symbol_sector", symbol, {"sector": sector})
                so_da_ghi_de += 1
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    logger.warning("Bỏ qua mã %s do lỗi liên tục: %s", symbol, exc)
                    break
                logger.warning(
                    "Kết nối bị ngắt khi ghi mã %s -> đang kết nối lại...", symbol
                )
                try:
                    storage.close()
                except Exception:  # noqa: BLE001
                    pass
                storage = Storage(db_path=db_path)
    logger.info("Đã ghi đè lại sector chuẩn cho %d mã.", so_da_ghi_de)

    # --- Bước 2: xóa mọi mã KHÔNG còn trong danh sách hiện tại ---
    all_keys_in_db = storage.query_all_keys("symbol_sector")
    orphan_symbols = [s for s in all_keys_in_db if s not in watchlist_symbols]

    logger.info("Tìm thấy %d mã KHÔNG còn trong watchlist hiện tại -> sẽ xóa.", len(orphan_symbols))
    so_da_xoa = 0

    for i, symbol in enumerate(orphan_symbols, start=1):
        # Thử xóa, nếu kết nối bị ngắt giữa chừng (phiên chạy dài, Supabase
        # tự đóng kết nối nhàn rỗi) -> tự MỞ LẠI kết nối mới rồi thử lại
        # đúng 1 lần cho mã đó, không dừng cả vòng lặp giữa chừng.
        for attempt in (1, 2):
            try:
                so_ban_ghi = storage.delete_key("symbol_sector", symbol)
                if so_ban_ghi:
                    so_da_xoa += 1
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    logger.warning("Bỏ qua mã %s do lỗi liên tục: %s", symbol, exc)
                    break
                logger.warning(
                    "Kết nối bị ngắt khi xóa mã %s -> đang kết nối lại...", symbol
                )
                try:
                    storage.close()
                except Exception:  # noqa: BLE001
                    pass
                storage = Storage(db_path=db_path)

        if i % 100 == 0:
            logger.info("Đã xử lý %d/%d mã orphan...", i, len(orphan_symbols))

    logger.info("Đã xóa dữ liệu ngành của %d mã orphan (không còn theo dõi).", so_da_xoa)
    storage.close()
    print(
        f"\nHoàn tất: ghi đè {so_da_ghi_de} mã đang theo dõi, "
        f"xóa {so_da_xoa}/{len(orphan_symbols)} mã cũ không còn liên quan.\n"
        "Vào lại dashboard -> mục 'Giai đoạn thị trường hiện tại' để kiểm tra."
    )


if __name__ == "__main__":
    main()
