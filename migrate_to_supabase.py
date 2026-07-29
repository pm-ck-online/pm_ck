"""
migrate_to_supabase.py
=========================
Di chuyển TOÀN BỘ dữ liệu đã có trong SQLite cục bộ (`./data/pm_ck.db`)
lên Supabase — chạy 1 LẦN DUY NHẤT khi chuyển sang dùng chung nhiều máy.

Yêu cầu trước khi chạy:
    1. Đã chạy `test_supabase_connection.py` thành công.
    2. `./data/pm_ck.db` đang có dữ liệu (đã chạy `main.py`/`run_full_market.py`
       ít nhất 1 lần trước đó).

Cách chạy:
    python migrate_to_supabase.py "postgresql://...connection string..."

Script này AN TOÀN để chạy lại nhiều lần — mỗi bản ghi di chuyển sẽ tạo
một bản ghi MỚI trên Supabase (giữ đúng lịch sử/timestamp gốc), không xóa
gì trên Supabase trước khi chạy. Nếu chạy lại, dữ liệu sẽ bị TRÙNG LẶP —
chỉ nên chạy 1 lần, hoặc xóa sạch bảng `records` trên Supabase trước khi
chạy lại.
"""

import sys
from datetime import datetime


def main() -> None:
    if len(sys.argv) < 2:
        print("Cách dùng: python migrate_to_supabase.py \"<connection_string>\"")
        sys.exit(1)

    supabase_connection_string = sys.argv[1]

    from core.storage import Storage

    print("=" * 70)
    print("DI CHUYỂN DỮ LIỆU: SQLite cục bộ -> Supabase")
    print("=" * 70)

    local_storage = Storage(db_path="./data/pm_ck.db")
    categories = local_storage.query_all_categories()

    if not categories:
        print("⚠️  Không tìm thấy dữ liệu nào trong './data/pm_ck.db'. Không có gì để di chuyển.")
        local_storage.close()
        sys.exit(0)

    print(f"Tìm thấy {len(categories)} nhóm dữ liệu (category) cần di chuyển:")
    for c in categories:
        print(f"  - {c}")

    confirm = input("\nBắt đầu di chuyển lên Supabase? (gõ 'yes' để xác nhận): ")
    if confirm.strip().lower() != "yes":
        print("Đã hủy.")
        local_storage.close()
        sys.exit(0)

    print("\nĐang kết nối Supabase...")
    remote_storage = Storage(db_path=supabase_connection_string)

    total_migrated = 0
    for category in categories:
        keys = local_storage.query_all_keys(category)
        for key in keys:
            history = local_storage.get_history(category, key, limit=100000)
            # get_history trả về MỚI NHẤT TRƯỚC -> đảo lại để giữ đúng thứ
            # tự thời gian khi ghi lại lên Supabase (không bắt buộc vì mỗi
            # bản ghi có timestamp riêng, nhưng ghi theo đúng thứ tự cho dễ
            # theo dõi tiến trình).
            for record in reversed(history):
                timestamp = datetime.fromisoformat(record["timestamp"])
                remote_storage.save(category, key, record["data"], timestamp=timestamp)
                total_migrated += 1

        print(f"  ✅ Đã di chuyển category '{category}' ({len(keys)} key).")

    local_storage.close()
    remote_storage.close()

    print("\n" + "=" * 70)
    print(f"🎉 HOÀN TẤT! Đã di chuyển tổng cộng {total_migrated} bản ghi lên Supabase.")
    print("=" * 70)
    print("\nBước tiếp theo: cập nhật config.yaml -> storage.path = connection string trên.")


if __name__ == "__main__":
    main()
