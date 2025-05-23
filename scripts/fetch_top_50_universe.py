import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Load API keys from .env
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
}

# Step 1: Get tradable US symbols from Alpaca
print("🔍 Fetching tradable symbols...")
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
assets = trading_client.get_all_assets()
symbols = [
    a.symbol for a in assets if a.tradable and a.exchange in {"NASDAQ", "NYSE", "ARCA"}
]
print(f"✅ Found {len(symbols)} tradable symbols")

# Step 2: Set time window for OHLCV
end_date = datetime.today()
start_date = end_date - timedelta(days=30)

params_base = {
    "timeframe": "1Day",
    "start": start_date.strftime("%Y-%m-%d"),
    "end": end_date.strftime("%Y-%m-%d"),
    "adjustment": "raw",
    "limit": 10000,
    "feed": "iex",  # Use IEX to avoid SIP subscription errors
}


def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


# Step 3: Fetch OHLCV data in chunks with a progress bar
symbol_batches = list(chunked(symbols, 200))  # max 200 symbols per request
all_rows = []

with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("📦 Fetching OHLCV...", total=len(symbol_batches))

    for batch in symbol_batches:
        params = params_base.copy()
        params["symbols"] = ",".join(batch)

        response = requests.get(BASE_URL, headers=HEADERS, params=params)
        if response.status_code != 200:
            progress.console.print(
                f"[red]❌ Error {response.status_code}[/red]: {response.text}"
            )
            progress.advance(task)
            continue

        data = response.json().get("bars", {})
        for symbol, bars in data.items():
            df = pd.DataFrame(bars)
            if df.empty:
                continue
            df["t"] = pd.to_datetime(df["t"])
            df.set_index("t", inplace=True)
            avg_volume = df["v"].mean()
            latest_close = df["c"].iloc[-1]
            all_rows.append(
                {
                    "symbol": symbol,
                    "avg_volume": avg_volume,
                    "price": latest_close,
                }
            )

        time.sleep(1.2)
        progress.advance(task)

# Step 4: Filter and export
df = pd.DataFrame(all_rows)
df = df[(df["price"] > 5) & (df["avg_volume"] > 500_000)]

# Final filter and sort
if not df.empty and "avg_volume" in df.columns:
    df_top50 = df.sort_values(by="avg_volume", ascending=False).head(50)
else:
    print("⚠️ No valid rows after filtering.")
    df_top50 = pd.DataFrame(columns=["symbol"])

# Save CSV
os.makedirs("universe", exist_ok=True)
df_top50["symbol"].to_csv("universe/top_50_by_volume.csv", index=False)

# Proper rich console print

console = Console()
console.print(
    "✅ Saved top 50 tickers to [bold green]universe/top_50_by_volume.csv[/bold green]"
)
