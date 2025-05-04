from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


class RequestDataClient:
    def __init__(self, data_dir: str = "data/ohlcv/minute/hive_parquet"):
        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect()

    def get_historical_data(
        self, symbol: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        query = f"""
        SELECT * FROM read_parquet('{self.data_dir}/symbol={symbol}/*.parquet')
        WHERE date BETWEEN '{start.date()}' AND '{end.date()}'
        ORDER BY timestamp
        """
        df = self.conn.execute(query).fetchdf()
        return df

    def get_symbols(self) -> list[str]:
        query = (
            f"SELECT DISTINCT symbol FROM read_parquet('{self.data_dir}/*/*.parquet')"
        )
        df = self.conn.execute(query).fetchdf()
        return df["symbol"].dropna().unique().tolist()

    def close(self):
        self.conn.close()
