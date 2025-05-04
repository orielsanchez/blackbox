from datetime import datetime

from trader.backtest.backtest_config import BacktestConfig
from trader.backtest.backtester import Backtester
from trader.core.engine import ModelBundle
from trader.data.parquet_data_loader import ParquetDataLoader
from trader.models.alpha.momentum import MomentumAlphaModel
from trader.models.execution.simple import SimpleExecutionModel
from trader.models.portfolio.equal_weight import EqualWeightPortfolioModel
from trader.models.risk.ewma import EWMARiskModel
from trader.models.tx_cost.fixed import FixedCostModel
from trader.utils.metrics import calculate_performance


def main():
    # Set up config
    config = BacktestConfig(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2023, 1, 1),
        initial_capital=100_000,
        warmup_period=20,
        top_n=50,
        backtest_id="momentum_top50",
    )

    # Set up model bundle
    models = ModelBundle(
        alpha=MomentumAlphaModel(),
        risk=EWMARiskModel(),
        tx_cost=FixedCostModel(),
        portfolio=EqualWeightPortfolioModel(top_n=config.top_n),
        execution=SimpleExecutionModel(),
    )

    # Load data
    data_loader = ParquetDataLoader()
    all_symbols = data_loader.get_available_symbols()
    selected_symbols = all_symbols[:2000]
    assert config.start_date and config.end_date, "Start and end dates must be set"
    # data = data_loader.load_all_data(config.start_date, config.end_date)
    data = data_loader.load_symbols(
        selected_symbols, config.start_date, config.end_date
    )

    # Run backtest
    backtester = Backtester(models, config)
    equity_curve = backtester.run(data)

    # Compute metrics
    metrics = calculate_performance(equity_curve)
    print("\nPerformance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
