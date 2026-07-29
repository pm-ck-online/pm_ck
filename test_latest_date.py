from vnstock import Market
import pandas as pd

mkt = Market()
df = mkt.equity("HPG").ohlcv(interval="1D", count=10)
print(df)
print("\nGiá đóng cửa mới nhất (HPG):", df["close"].iloc[-1])