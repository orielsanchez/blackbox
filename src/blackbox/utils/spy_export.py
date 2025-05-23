from pathlib import Path

import pandas as pd

from blackbox.data.ohlcv_loader import OHLCVDataLoader


def export_spy_benchmark(
    db_path: str,
    output_csv: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    symbol: str = "SPY",
    table: str = "daily_data",
) -> None:
    """Query OHLCV DuckDB and export SPY daily close CSV."""
    loader = OHLCVDataLoader(db_path=db_path, table=table)
    df = loader.load_ohlcv_raw(start_date, end_date, symbols=[symbol])

    # Extract date + close
    spy = (
        df.loc[(slice(None), symbol), ["close"]]
        .reset_index(level="symbol", drop=True)
        .reset_index()
        .sort_values("date")
    )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    spy.to_csv(output_csv, index=False)
    print(f"✅ Exported SPY benchmark to: {output_csv}")
