import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from trader.backtest.backtest_config import BacktestConfig
from trader.backtest.backtester import Backtester
from trader.core.engine import ModelBundle
from trader.data.duckdb_loader import DuckDBDailyLoader
from trader.models.alpha.momentum import MomentumAlphaModel
from trader.models.execution.simple import SimpleExecutionModel
from trader.models.portfolio.equal_weight_score import \
    EqualWeightScorePortfolioModel
from trader.models.risk.ewma import EWMARiskModel
from trader.models.slippage.percent import PercentSlippageModel
from trader.models.tx_cost.fixed import FixedCostModel
from trader.utils.metrics import (calculate_performance,
                                  plot_log_equity_with_regression)

# === Configure logging ===
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_universe(path: str, limit: int | None = None) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"Universe file missing 'symbol' column: {path}")
    symbols = df["symbol"].dropna().astype(str).tolist()
    return symbols if limit is None else symbols[:limit]


def save_backtest_results(
    backtest_id: str, result_df: pd.DataFrame, equity_curve: pd.Series, metrics: dict
) -> None:
    outdir = Path(f"results/{backtest_id}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Save trades
    result_path = outdir / "trades.parquet"
    result_df.to_parquet(result_path)
    logger.info(f"💾 Saved trades to {result_path}")

    # Save equity curve
    equity_path = outdir / "equity.parquet"
    equity_curve.to_frame("equity").to_parquet(equity_path)
    logger.info(f"💾 Saved equity curve to {equity_path}")

    # Save metrics
    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"💾 Saved performance metrics to {metrics_path}")


def main():
    # === CLI Args ===
    parser = argparse.ArgumentParser(description="Run backtest on universe")
    parser.add_argument(
        "--universe",
        type=str,
        default="universe/universe.csv",
        help="Path to universe CSV file (with 'symbol' column)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of symbols to use",
    )
    args = parser.parse_args()

    # === Time window ===
    end_date = datetime(2023, 12, 31)
    start_date = end_date - timedelta(days=365)

    # === Load symbols & data ===
    universe = load_universe(args.universe, args.limit)
    loader = DuckDBDailyLoader("db/ohlcv.duckdb")
    data = loader.load_symbols(universe, start=start_date, end=end_date)
    loader.close()

    if data.empty:
        logger.warning("⚠️  No data loaded. Exiting.")
        return

    # === Model setup ===
    models = ModelBundle(
        alpha=MomentumAlphaModel(),
        risk=EWMARiskModel(span=20),
        tx_cost=FixedCostModel(cost_per_share=0.01),
        slippage=PercentSlippageModel(slippage_pct=0.001),
        portfolio=EqualWeightScorePortfolioModel(
            max_positions=10,
            min_price=2.0,
            min_volume=1_000_000,
            max_notional_pct=0.10,
        ),
        execution=SimpleExecutionModel(),
    )

    config = BacktestConfig(
        initial_capital=100_000,
        start_date=start_date,
        end_date=end_date,
        warmup_period=30,
        backtest_id="ewma-momentum-daily",
        allow_shorting=False,
        settlement_delay=0,
    )

    # === Run backtest ===
    backtester = Backtester(models=models, config=config)
    result_df, equity_curve = backtester.run(data)

    # === Summary output ===
    logger.info("✅ Backtest complete.")
    print("\n📊 Backtest Summary:")
    metrics = calculate_performance(equity_curve)

    plot_log_equity_with_regression(equity_curve)
    for key, val in metrics.items():
        print(f"{key:<30} {val}")

    # === Save results to disk ===
    save_backtest_results(config.backtest_id, result_df, equity_curve, metrics)


if __name__ == "__main__":
    main()
