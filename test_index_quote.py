from vnstock import Market

mkt = Market()

print("=== Quote cho CỔ PHIẾU (HPG) — đối chiếu, biết là đang chạy đúng ===")
df_stock = mkt.quote("HPG")
print(df_stock.columns.tolist())
print(df_stock)

print("\n=== Quote cho CHỈ SỐ (VN100) — đang lỗi ===")
try:
    df_index = mkt.quote("VN100")
    print(df_index.columns.tolist())
    print(df_index)
except Exception as e:
    print("LỖI:", e)

print("\n=== Thử luôn VNINDEX và VN30 để biết có đồng nhất không ===")
for sym in ["VNINDEX", "VN30"]:
    try:
        df = mkt.quote(sym)
        print(f"--- {sym} ---")
        print(df.columns.tolist())
        print(df)
    except Exception as e:
        print(f"--- {sym} --- LỖI:", e)