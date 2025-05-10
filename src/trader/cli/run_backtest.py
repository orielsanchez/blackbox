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
from trader.models.alpha.composite_alpha import CompositeAlphaModel
from trader.models.alpha.mean_reversion import MeanReversionAlphaModel
from trader.models.alpha.momentum import MomentumAlphaModel
from trader.models.alpha.regime_aware import RegimeAwareAlphaModel
from trader.models.execution.simple import SimpleExecutionModel
from trader.models.portfolio.equal_weight_score import \
    EqualWeightScorePortfolioModel
from trader.models.portfolio.volatility_targeted_score import \
    VolatilityTargetedScorePortfolioModel
from trader.models.portfolio.weighted_score import WeightedScorePortfolioModel
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
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=365 * 4)

    # === Load symbols & data ===
    universe = load_universe(args.universe, args.limit)
    loader = DuckDBDailyLoader("db/ohlcv.duckdb")
    data = loader.load_symbols(universe, start=start_date, end=end_date)
    loader.close()

    if data.empty:
        logger.warning("⚠️  No data loaded. Exiting.")
        return

    alpha_model = CompositeAlphaModel(
        models=[MomentumAlphaModel(lookback=10), MeanReversionAlphaModel(window=10)],
        weights=[0.5, 0.5],
    )

    alpha_model = RegimeAwareAlphaModel(
        momentum_model=MomentumAlphaModel(lookback=10),
        mean_rev_model=MeanReversionAlphaModel(window=5),
        window=30,
        trend_threshold=0.6,
    )
    models = ModelBundle(
        alpha=alpha_model,
        risk=EWMARiskModel(span=10, min_span=3, strict=False),
        tx_cost=FixedCostModel(cost_per_share=0.01),
        slippage=PercentSlippageModel(slippage_pct=0.001),
        portfolio=VolatilityTargetedScorePortfolioModel(
            weights={
                "alpha_score": 0.7,
                "risk_score": -0.2,  # still used to penalize score
                "tx_cost": -0.05,
                "slippage": -0.05,
            },
            min_alpha=0.01,
            top_n=50,
            volatility_column="risk_score",  # assumes risk_score = EWMA volatility
            volatility_targeting=True,
        ),
        execution=SimpleExecutionModel(),
    )

    config = BacktestConfig(
        initial_capital=1_000,
        start_date=start_date,
        end_date=end_date,
        warmup_period=30,
        backtest_id="ewma-momentum-daily",
        allow_shorting=False,
        settlement_delay=2,
    )

    # === Run backtest ===
    backtester = Backtester(models=models, config=config)
    result_df, equity_curve = backtester.run(data)

    # === Summary output ===
    logger.info("✅ Backtest complete.")
    print("\n📊 Backtest Summary:")
    metrics = calculate_performance(equity_curve)

    for key, val in metrics.items():
        print(f"{key:<30} {val}")

    plot_log_equity_with_regression(equity_curve)
    # === Save results to disk ===
    save_backtest_results(config.backtest_id, result_df, equity_curve, metrics)


if __name__ == "__main__":
    main()
