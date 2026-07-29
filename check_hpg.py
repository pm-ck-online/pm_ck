from core.storage import Storage

storage = Storage(db_path="./data/pm_ck.db")
record = storage.get_latest("ohlcv_history", "HPG")

if record is None:
    print("KHÔNG có dữ liệu ohlcv_history cho HPG trong storage!")
else:
    print("Thời điểm lưu:", record["timestamp"])
    records = record["data"]["records"]
    print("Số phiên:", len(records))
    print("5 phiên cuối cùng:")
    for r in records[-5:]:
        print(r)

storage.close()