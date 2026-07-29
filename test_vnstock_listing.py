from vnstock import Listing

listing = Listing(source="KBS")

print("=== Toàn bộ mã chứng khoán ===")
df = listing.all_symbols(to_df=True)
print(df.columns.tolist())
print(df.head())
print(f"Tổng số mã: {len(df)}")

print("\n=== Mã theo ngành ===")
df_industry = listing.symbols_by_industries()
print(type(df_industry))
print(df_industry.columns.tolist() if hasattr(df_industry, "columns") else df_industry[:10])