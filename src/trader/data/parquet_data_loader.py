import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ParquetDataLoader:
    def __init__(
        self,
        data_dir: str = "data/ohlcv/minute/hive_parquet",
        engine: str = "pyarrow",
    ):
        self.data_dir = Path(data_dir).resolve()
        self.engine = engine
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        logger.info(f"Initialized ParquetDataLoader with data path: {self.data_dir}")

    def _load_single_symbol(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        symbol_dir = self.data_dir / f"symbol={symbol}"
        if not symbol_dir.exists():
            tqdm.write(f"⚠️  Directory not found for symbol {symbol}")
            return pd.DataFrame()

        # Collect relevant files
        relevant_files = []
        for date_dir in symbol_dir.glob("date=*"):
            try:
                date_str = date_dir.name.split("=")[-1]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if not (start <= file_date <= end):
                    continue
                relevant_files.extend(date_dir.glob("part-*.parquet"))
            except ValueError:
                tqdm.write(f"⚠️  Skipping unrecognized date format in {date_dir}")
                continue

        dfs = []
        for f in relevant_files:
            try:
                df = pd.read_parquet(f, engine=self.engine)  # type: ignore[arg-type]

                if "window_start" in df.columns:
                    df = df.rename(columns={"window_start": "timestamp"})

                if "timestamp" not in df.columns:
                    tqdm.write(f"⚠️  No timestamp column found in {f}, skipping")
                    continue

                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df = df.dropna(subset=["timestamp"])

                # Standardize to price column
                if "close" in df.columns and "price" not in df.columns:
                    df = df.rename(columns={"close": "price"})

                df["symbol"] = symbol
                dfs.append(df)
            except Exception as e:
                tqdm.write(f"❌ Error loading {f} for {symbol}: {e}")

        if not dfs:
            return pd.DataFrame()

        symbol_df = pd.concat(dfs, ignore_index=True)
        symbol_df = symbol_df.drop_duplicates(subset=["symbol", "timestamp"])
        symbol_df = symbol_df.sort_values(["symbol", "timestamp"])
        return symbol_df

    def load_symbols(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        max_workers: int = 8,
    ) -> pd.DataFrame:
        logger.info(
            f"Loading {len(symbols)} symbols from {start.date()} to {end.date()}"
        )
        all_data = []
        max_workers = min(max_workers, len(symbols))

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._load_single_symbol, symbol, start, end
                    ): symbol
                    for symbol in symbols
                }

                for i, future in enumerate(
                    tqdm(as_completed(futures), total=len(futures), file=sys.stdout)
                ):
                    symbol = futures[future]
                    try:
                        df = future.result()
                        all_data.append(df)
                        tqdm.write(
                            f"[{i+1}/{len(symbols)}] ✅ Loaded {len(df)} rows for {symbol}"
                        )
                    except Exception as e:
                        tqdm.write(
                            f"[{i+1}/{len(symbols)}] ❌ Error processing {symbol}: {e}"
                        )
        except KeyboardInterrupt:
            tqdm.write("⛔ KeyboardInterrupt: Loader shutting down early.")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

        if not all_data:
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=["symbol", "timestamp"])
        df = df.sort_values(["symbol", "timestamp"])
        return df

    def close(self):
        logger.info("No database connection to close (using native Parquet reads).")
