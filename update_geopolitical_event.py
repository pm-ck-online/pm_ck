"""
update_geopolitical_event.py
==============================
Cập nhật NHANH trạng thái "sự kiện địa chính trị" dựa trên tin tức thời
sự đã tra cứu (Claude tìm và phân loại ngày 24/07/2026) — chạy 1 lần để
ghi thẳng vào storage, không cần thao tác tay trên dashboard.

Nguồn tham khảo:
    - VinaCapital, "Iran War: Modest Impact on Vietnam" (10/07/2026)
    - Tuổi Trẻ, "VN-Index plummeted nearly 120 points in three sessions" (22/07/2026)
    - Vietstock, "Selling force pushes VN-Index below 1,700 points" (23/07/2026)

Bạn có thể chạy lại script này bất cứ khi nào muốn cập nhật lại theo tin
tức mới hơn — chỉ cần sửa `EVENT_KEY` và `NOTE` bên dưới.
"""

from datetime import date

import yaml

from core.storage import Storage

# ==============================================================================
# SỬA 2 DÒNG NÀY MỖI KHI MUỐN CẬP NHẬT LẠI THEO TÌNH HÌNH MỚI
# ==============================================================================
EVENT_KEY = "conflict_outbreak"  # xem core/macro_score_engine.py để biết các mức hợp lệ
NOTE = (
    "Xung đột Iran (Mỹ/Israel-Iran) nổ ra từ đầu T3/2026, vẫn tiếp diễn tới "
    "T7/2026 (VinaCapital gọi là 'Iran War', 10/07/2026). Giá dầu tăng mạnh do "
    "lo ngại đóng eo biển Hormuz. VN-Index giảm gần 120 điểm chỉ trong 3 phiên "
    "(22-23/07/2026), xuống dưới 1.700 điểm — kết hợp thêm rủi ro thuế quan Mỹ "
    "mới với hàng Việt Nam và bán ròng khối ngoại. "
    "Nguồn: VinaCapital, Tuổi Trẻ, Vietstock (tra cứu 24/07/2026)."
)
# ==============================================================================


def main() -> None:
    with open("config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    storage = Storage(db_path=config["storage"]["path"])
    storage.save("manual_macro_setting", "geopolitical_event", {
        "event_key": EVENT_KEY,
        "note": NOTE,
        "updated_date": date.today().isoformat(),
    })
    storage.close()

    print(f"Đã cập nhật trạng thái sự kiện địa chính trị: {EVENT_KEY}")
    print(f"Ghi chú: {NOTE}")


if __name__ == "__main__":
    main()
