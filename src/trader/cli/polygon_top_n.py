import os

import pandas as pd
import requests
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    raise EnvironmentError("POLYGON_API_KEY not found in .env")

SNAPSHOT_URL = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
params = {"apiKey": API_KEY}

response = requests.get(SNAPSHOT_URL, params=params)
data = response.json()

if "tickers" not in data:
    raise ValueError(f"Unexpected response format: {data}")

# Sort tickers by volume descending
sorted_tickers = sorted(data["tickers"], key=lambda x: x["day"]["v"], reverse=True)

# Get only the top 2000 symbols
top_2000_symbols = [item["ticker"] for item in sorted_tickers[:2000]]

# Save to CSV
df = pd.DataFrame(top_2000_symbols, columns=["symbol"])
df.to_csv("top_2000_by_volume.csv", index=False)

print("Saved top 2000 symbols by volume to top_2000_by_volume.csv")
