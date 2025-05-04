from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


class ParquetDataLoader:
    def __init__(self, data_dir: str = "data/ohlcv/minute/hive_parquet"):
        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect()

    def load_all_data(self, start: datetime, end: datetime) -> pd.DataFrame:
        query = f"""
        SELECT * FROM read_parquet('{self.data_dir}/symbol=*/date=*/part-*.parquet')
        WHERE date BETWEEN '{start.date()}' AND '{end.date()}'
        ORDER BY timestamp
        """
        df = self.conn.execute(query).fetchdf()
        return df

    def load_symbols(
        self, symbols: list[str], start: datetime, end: datetime
    ) -> pd.DataFrame:
        symbol_filter = ", ".join(f"'{s}'" for s in symbols)
        query = f"""
        SELECT * FROM read_parquet('{self.data_dir}/symbol=*/date=*/part-*.parquet')
        WHERE date BETWEEN '{start.date()}' AND '{end.date()}'
        AND symbol IN ({symbol_filter})
        ORDER BY timestamp
        """
        df = self.conn.execute(query).fetchdf()
        return df

    def get_available_symbols(self) -> list[str]:
        query = f"SELECT DISTINCT symbol from read_parquet('{self.data_dir}/symbol=*/date=*/part-*.parquet')"
        df = self.conn.execute(query).fetchdf()
        return df["symbol"].dropna().unique().tolist()

    def close(self):
        self.conn.close()
