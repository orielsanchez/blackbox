import argparse

from blackbox.config.loader import load_config
from blackbox.tuning.grid_search import run_grid_search
from blackbox.utils.io import load_yaml
from blackbox.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter grid search.")
    parser.add_argument("--config", required=True, help="Path to base strategy config YAML.")
    parser.add_argument("--search-space", required=True, help="Path to search_space YAML.")
    parser.add_argument(
        "--metric",
        default="sharpe",
        help="Metric to optimize (e.g. sharpe, return, ic_mean).",
    )
    parser.add_argument(
        "--results-dir",
        default="results/tuning",
        help="Directory to save grid results.",
    )

    args = parser.parse_args()

    # Load and setup config + logger
    config = load_config(args.config)
    setup_logger(config)

    # Load tuning search space
    search_space = load_yaml(args.search_space)

    # Run the grid search
    run_grid_search(
        config_path=args.config,
        search_space=search_space,
        metric=args.metric,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
