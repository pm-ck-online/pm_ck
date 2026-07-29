"""
test_supabase_connection.py
==============================
Script KIỂM TRA kết nối tới Supabase — chạy 1 lần để xác nhận connection
string đúng TRƯỚC KHI đổi `config.yaml` sang dùng Supabase làm storage
chính thức.

Cách lấy connection string:
    1. Vào https://supabase.com/dashboard/project/ktlnkpelgdyxxfvfwujb
    2. Vào "Project Settings" (biểu tượng bánh răng) -> "Database"
    3. Mục "Connection string" -> chọn tab "URI"
    4. Copy chuỗi dạng: postgresql://postgres.xxxxx:[YOUR-PASSWORD]@aws-x-xx-xxxx-x.pooler.supabase.com:6543/postgres
    5. Thay [YOUR-PASSWORD] bằng mật khẩu database bạn đã đặt lúc tạo dự án
       (KHÔNG phải mật khẩu đăng nhập Supabase — là mật khẩu riêng cho
       database, có thể xem/đặt lại ở cùng trang Database Settings)

Chạy: python test_supabase_connection.py "postgresql://...connection string của bạn..."
"""

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Cách dùng: python test_supabase_connection.py \"<connection_string>\"")
        print(
            "\nLấy connection string tại: Supabase Dashboard -> Project "
            "Settings -> Database -> Connection string (chọn tab 'URI')"
        )
        sys.exit(1)

    connection_string = sys.argv[1]

    print("=" * 70)
    print("ĐANG KIỂM TRA KẾT NỐI TỚI SUPABASE...")
    print("=" * 70)

    from core.storage import Storage

    try:
        storage = Storage(db_path=connection_string)
        print("✅ Kết nối thành công! Đã tạo bảng 'records' (nếu chưa có).")
    except Exception as exc:
        print(f"❌ KẾT NỐI THẤT BẠI: {exc}")
        print(
            "\nKiểm tra lại: (1) connection string copy đúng chưa, (2) đã "
            "thay [YOUR-PASSWORD] bằng mật khẩu database thật chưa, (3) "
            "mạng máy tính có bị chặn cổng 5432/6543 không (một số mạng "
            "công ty/cơ quan có thể chặn)."
        )
        sys.exit(1)

    print("\nĐang thử ghi + đọc thử 1 bản ghi kiểm tra...")
    storage.save("_ket_noi_thu", "test", {"thong_diep": "Xin chào từ pm_ck!"})
    record = storage.get_latest("_ket_noi_thu", "test")
    print("✅ Ghi + đọc dữ liệu thành công:", record["data"])

    # Dọn dẹp bản ghi kiểm tra
    storage.delete_key("_ket_noi_thu", "test")
    storage.close()

    print("\n" + "=" * 70)
    print("🎉 MỌI THỨ HOẠT ĐỘNG ĐÚNG! Bạn có thể cập nhật config.yaml:")
    print("=" * 70)
    print(f'\nstorage:\n  path: "{connection_string}"\n')


if __name__ == "__main__":
    main()
