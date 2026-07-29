from vnstock import Market

mkt = Market()

print("=== OHLCV ===")
df = mkt.equity("HPG").ohlcv(start="2025-01-01", end="2026-01-01", interval="1D")
print(df.columns.tolist())
print(df.head())

print("\n=== Giá hiện tại (quote) ===")
price = mkt.quote("HPG")
print(type(price))
print(price)
print("\n=== Danh sách đầy đủ các cột ===")
print(price.columns.tolist())

print("\n=== Toàn bộ dữ liệu dòng đầu tiên ===")
print(price.iloc[0].to_dict())