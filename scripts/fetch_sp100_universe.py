import os

import pandas as pd

URL = "https://en.wikipedia.org/wiki/S%26P_100"

# Output path
os.makedirs("universe", exist_ok=True)
OUTPUT_PATH = "universe/sp100.csv"

# Load Wikipedia table
tables = pd.read_html(URL, match="Symbol")
df = tables[0]

# Extract and clean tickers
tickers = df["Symbol"].astype(str).str.strip().drop_duplicates()
tickers.to_csv(OUTPUT_PATH, index=False, header=["symbol"])

print(f"✅ Saved S&P 100 tickers to {OUTPUT_PATH}")
