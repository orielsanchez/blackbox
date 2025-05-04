from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class BacktestConfig:
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 100_000.0
    top_n: int = 10  # top-N symbols to trade per day
    warmup_period: int = 20  # number of bars to skip for indicator warmup
    save_results: bool = True
    results_dir: str = "results"
    output_format: str = "parquet"  # or 'json'
    backtest_id: Optional[str] = None  # auto-generated or user-defined
