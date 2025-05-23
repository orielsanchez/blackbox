import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from blackbox.config.loader import load_config
from blackbox.tuning.grid_tuner import GridTuner
from blackbox.tuning.optuna_tuner import OptunaTuner
from blackbox.tuning.tuned_walkforward_runner import TunedWalkforwardRunner
from blackbox.tuning.walkforward_plot import (
    load_all_test_curves,
    load_spy_benchmark,
    plot_combined_equity_curve,
)
from blackbox.utils.logger import setup_logger
from blackbox.utils.spy_export import export_spy_benchmark


def get_metric(metrics: dict, key: str):
    """Case-insensitive safe getter for metrics."""
    for k, v in metrics.items():
        if k.lower() == key.lower():
            return v
    return "N/A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuner",
        choices=["grid", "optuna"],
        default="optuna",
        help="Choose tuner type",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=9,
        help="Number of parallel jobs for optuna tuning (only applies if --tuner optuna)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Plot walkforward equity curve vs SPY",
    )
    parser.add_argument(
        "--config", type=str, default="experiments/strategies/regime_composite.yaml"
    )
    args = parser.parse_args()

    # ────── Load config and logger ──────
    config_path = Path(args.config)
    config = load_config(config_path)
    setup_logger(config)

    # ────── Define tuning space ──────
    grid_space = {
        "mean_reversion__rsi_period": [10, 14, 20],
        "mean_reversion__zscore_period": [5, 10],
        "mean_reversion__threshold": [0.2, 0.3, 0.4],
    }

    optuna_space = {
        "mean_reversion__rsi_period": [10, 20],  # interpreted as range
        "mean_reversion__zscore_period": [5, 15],
        "mean_reversion__threshold": [0.2, 0.5],
    }

    # ────── Initialize tuner ──────
    if args.tuner == "grid":
        tuner = GridTuner(search_space=grid_space)
    else:
        tuner = OptunaTuner(
            param_space=optuna_space,
            n_trials=30,
            direction="maximize",
            n_jobs=args.n_jobs,
        )

    # ────── Run walkforward validation ──────
    runner = TunedWalkforwardRunner(
        base_config_path=str(config_path),
        tuner=tuner,
        train_days=252,
        test_days=63,
        output_dir="tmp/tuned_walkforward_configs",
    )
    results = runner.run()

    # ────── Print summary ──────
    print("\n📊 Walkforward Results:")
    for r in results:
        run_id = r["run_id"]
        metrics = r["metrics"]
        sharpe = get_metric(metrics, "Sharpe Ratio")
        ret = get_metric(metrics, "Total Return (%)")
        dd = get_metric(metrics, "Max Drawdown (%)")
        ir = get_metric(metrics, "Information Ratio")
        print(f"{run_id} | Sharpe: {sharpe} | Return: {ret}% | MaxDD: {dd}% | IR: {ir}")

    # ────── Optional Plot ──────
    if args.plot:
        print("\n📈 Generating walkforward equity curve vs SPY...")
        test_curves = load_all_test_curves("backtests")
        start_date = test_curves["date"].min()
        end_date = test_curves["date"].max()

        export_spy_benchmark(
            db_path="db/ohlcv.duckdb",
            output_csv="data/spy.csv",
            start_date=start_date,
            end_date=end_date,
        )

        spy = load_spy_benchmark("data/spy.csv", start_date, end_date)
        plot_combined_equity_curve(test_curves, spy)


if __name__ == "__main__":
    main()
