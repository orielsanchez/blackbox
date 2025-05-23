from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.table import Table

from blackbox.config.loader import load_config
from blackbox.core.backtest.backtest_engine import run_backtest_loop
from blackbox.core.types.context import (
    BacktestConfig,
    BacktestMetrics,
    FeatureMatrixInfo,
    PositionTracker,
    PreparedDataBundle,
    StrategyContext,
)
from blackbox.data.ohlcv_updater import PolygonOHLCVUpdater
from blackbox.data.processing import prepare_data_bundle
from blackbox.feature_generators.utils import collect_all_feature_specs
from blackbox.models.factory import build_models
from blackbox.research.metrics import PerformanceMetrics
from blackbox.utils.io import write_results
from blackbox.utils.logger import RichLogger


def run_backtest(
    config_path: str,
    use_cached_features: bool = True,
    refresh_data: bool = False,
    plot_equity: bool = False,
    output_dir: Path = Path(),
) -> None:
    config: BacktestConfig = load_config(config_path)

    if not getattr(config, "run_id", None):
        config.run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"

    if output_dir == Path():
        output_dir = Path("backtests") / config.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = RichLogger(level=config.log_level)
    logger.info(f"📄 Loaded config from {config_path}")

    models = build_models(config)
    universe = pd.read_csv(config.universe_file)["symbol"].tolist()

    if refresh_data:
        logger.info("🌐 Refreshing OHLCV data from Polygon...")
        PolygonOHLCVUpdater(
            db_path=config.data.db_path,
            table="daily_data",
            universe_path=config.universe_file,
        ).run()

    feature_specs = collect_all_feature_specs(config)
    data = prepare_data_bundle(
        data_config=config.data,
        feature_specs=feature_specs,
        universe=universe,
        run_id=config.run_id,
        force_reload=not use_cached_features,
        config_start_date=config.start_date,
        config_end_date=config.end_date,
        logger=logger,
        output_dir=output_dir,
    )

    context = StrategyContext(
        config=config,
        logger=logger,
        models=models,
        tracker=PositionTracker(config.initial_portfolio_value),
        initial_equity=config.initial_portfolio_value,
    )

    # Trim to backtest range
    start, end = pd.to_datetime(config.start_date), pd.to_datetime(config.end_date)
    filtered_snapshots = [s for s in data.snapshots if start <= s.date <= end]
    feature_matrix = data.feature_matrix.loc[
        (data.feature_matrix.index.get_level_values("date") >= start)
        & (data.feature_matrix.index.get_level_values("date") <= end)
    ]
    valid_dates = [d for d in data.metadata.dates if start <= d <= end]

    data = PreparedDataBundle(
        snapshots=filtered_snapshots,
        feature_matrix=feature_matrix,
        metadata=FeatureMatrixInfo(
            features=data.metadata.features,
            dates=set(valid_dates),
            min_date=min(valid_dates) if valid_dates else start,
            warmup=data.metadata.warmup,
        ),
    )

    # 🧠 Run core loop
    logs, trade_records = run_backtest_loop(context, data)

    # 📊 Compute metrics
    metrics_calculator = PerformanceMetrics(
        initial_value=config.initial_portfolio_value,
        risk_free_rate=config.risk_free_rate,
    )
    metrics_dict, equity_curve = metrics_calculator.compute_metrics(
        pd.DataFrame([log.__dict__ for log in logs]), return_equity=True
    )
    metrics = BacktestMetrics(summary=metrics_dict, equity_curve=equity_curve)

    # 💾 Write all results (logs, trades, equity, metrics, config)
    write_results(
        logs=logs,
        metrics=metrics,
        config=config,
        output_dir=output_dir,
        equity_curve=equity_curve,
        plot_equity=plot_equity,
    )

    # 🧾 Optionally write trades.csv
    if trade_records:
        trades_path = output_dir / "trades.csv"
        df_trades = pd.DataFrame(trade_records)
        df_trades.sort_values(by=["date", "symbol"], inplace=True)
        df_trades.to_csv(trades_path, index=False)
        logger.info(f"📄 Saved trades to {trades_path}")

    # 📋 Summary table
    table = Table(title="Backtest Summary", show_edge=True, header_style="bold cyan")
    table.add_column("Metric", style="dim", no_wrap=True)
    table.add_column("Value", justify="right")
    for key, value in metrics.summary.items():
        value_str = f"{value:,.4f}" if isinstance(value, float) else str(value)
        table.add_row(key.replace("_", " ").title(), value_str)
    logger.console.print(table)

    logger.info(f"🏁 Backtest complete — results saved to {output_dir}")
