import argparse
from pathlib import Path

from blackbox.config.loader import load_config
from blackbox.tuning.grid_tuner import GridTuner
from blackbox.utils.io import load_yaml
from blackbox.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter grid search.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to base strategy config YAML file.",
    )
    parser.add_argument(
        "--search-space",
        required=True,
        help="Path to search space YAML file defining parameter grid.",
    )
    parser.add_argument(
        "--metric",
        default="sharpe",
        help="Metric to optimize (e.g. sharpe, return, ic_mean).",
    )
    parser.add_argument(
        "--results-dir",
        default="results/tuning",
        help="Directory to save tuning results and plots.",
    )

    args = parser.parse_args()

    # Load config and setup logging
    config = load_config(args.config)
    setup_logger(config)

    # Load search space
    search_space = load_yaml(args.search_space)

    # Run grid tuner
    tuner = GridTuner(search_space=search_space)
    tuner.tune(
        config_path=Path(args.config),
        metric=args.metric,
        results_dir=Path(args.results_dir),
    )


if __name__ == "__main__":
    main()
