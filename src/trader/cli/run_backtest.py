import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from trader.backtest.backtest_config import BacktestConfig
from trader.backtest.backtester import Backtester
from trader.core.engine import ModelBundle
from trader.data.parquet_data_loader import ParquetDataLoader
from trader.models.alpha.momentum import MomentumAlphaModel
from trader.models.execution.simple import SimpleExecutionModel
from trader.models.portfolio.equal_weight import EqualWeightPortfolioModel
from trader.models.risk.ewma import EWMARiskModel
from trader.models.slippage.percent import PercentSlippageModel
from trader.models.tx_cost.fixed import FixedCostModel
from trader.utils.metrics import calculate_performance

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_universe(path: str = "universe/top_2000_by_volume.csv") -> list[str]:
    df = pd.read_csv(path)
    return df["symbol"].dropna().tolist()


def main():
    # Load stock universe
    universe = load_universe()
    logger.info(f"Loaded {len(universe)} symbols from universe file")

    # Date range for backtest
    end_date = datetime(2023, 12, 31)
    start_date = end_date - timedelta(days=365 * 2)

    # Load data
    loader = ParquetDataLoader()
    data = loader.load_symbols(universe, start=start_date, end=end_date)
    loader.close()

    # Configure models
    models = ModelBundle(
        alpha=MomentumAlphaModel(),
        risk=EWMARiskModel(span=20),
        tx_cost=FixedCostModel(cost_per_share=0.01),
        portfolio=EqualWeightPortfolioModel(top_n=10),
        execution=SimpleExecutionModel(
            slippage_model=PercentSlippageModel(slippage_rate=0.001)
        ),
    )

    # Backtest config
    config = BacktestConfig(
        initial_capital=100_000,
        start_date=start_date,
        end_date=end_date,
        warmup_period=30,
        backtest_id="top2000-ewma-momentum",
    )

    # Run backtest
    backtester = Backtester(models=models, config=config)
    equity_curve = backtester.run(data)

    # Report
    metrics = calculate_performance(equity_curve)
    print("\nBacktest Summary:")
    for key, val in metrics.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
