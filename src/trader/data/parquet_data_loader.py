import json
import logging
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

SYMBOL_CACHE_FILE = Path(".symbol_cache.json")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ParquetDataLoader:
    def __init__(self, data_dir: str = "data/ohlcv/minute/hive_parquet"):
        self.data_dir = Path(data_dir).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self.conn = duckdb.connect()
        logger.info(f"Initialized ParquetDataLoader with data path: {self.data_dir}")

    def load_all_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        logger.info(f"Loading all data from {start.date()} to {end.date()}")
        query = f"""
        SELECT * FROM read_parquet('{self.data_dir}/symbol=*/date=*/part-*.parquet')
        WHERE date BETWEEN '{start.date()}' AND '{end.date()}'
        ORDER BY timestamp
        """
        df = self.conn.execute(query).fetchdf()
        logger.info(f"Loaded {len(df)} rows")
        return df

    def load_symbols(
        self, symbols: list[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        logger.info(
            f"Loading data for {len(symbols)} symbols from {start.date()} to {end.date()}"
        )
        symbol_filter = ", ".join(f"'{s}'" for s in symbols)
        query = f"""
        SELECT * FROM read_parquet('{self.data_dir}/symbol=*/date=*/part-*.parquet')
        WHERE date BETWEEN '{start.date()}' AND '{end.date()}'
        AND symbol IN ({symbol_filter})
        ORDER BY timestamp
        """
        df = self.conn.execute(query).fetchdf()
        logger.info(f"Loaded {len(df)} rows for requested symbols")
        return df

    def get_available_symbols(self, use_cache: bool = True) -> list[str]:
        if use_cache and SYMBOL_CACHE_FILE.exists():
            logger.info("Loading symbols from cache...")
            with open(SYMBOL_CACHE_FILE, "r") as f:
                return json.load(f)

        logger.info("Querying symbols from Parquet files...")
        query = f"""
        SELECT DISTINCT symbol
        FROM read_parquet('{self.data_dir}/symbol=*/date=*/part-*.parquet')
        """
        df = self.conn.execute(query).fetchdf()
        symbols = df["symbol"].dropna().unique().tolist()
        logger.info(f"Found {len(symbols)} unique symbols")

        if use_cache:
            with open(SYMBOL_CACHE_FILE, "w") as f:
                json.dump(symbols, f)
            logger.info(f"Cached {len(symbols)} symbols to {SYMBOL_CACHE_FILE}")

        return symbols

    def close(self):
        self.conn.close()
        logger.info("Closed DuckDB connection")
