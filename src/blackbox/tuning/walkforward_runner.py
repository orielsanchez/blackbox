from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from blackbox.core.runner import run_backtest
from blackbox.research.metrics import load_metrics_for_run
from blackbox.tuning.walkforward import generate_walkforward_windows
from blackbox.utils.io import load_yaml, save_yaml


class WalkforwardRunner:
    def __init__(
        self,
        base_config_path: str,
        train_days: int,
        test_days: int,
        output_dir: str = "tmp/walkforward_configs",
    ):
        self.base_config_path = Path(base_config_path)
        self.train_days = train_days
        self.test_days = test_days
        self.output_dir = Path(output_dir)
        self.base_config = load_yaml(base_config_path)
        self.start_date = self.base_config["start_date"]
        self.end_date = self.base_config["end_date"]

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[Dict[str, Any]]:
        results = []
        windows = generate_walkforward_windows(
            self.start_date, self.end_date, self.train_days, self.test_days
        )

        for _, test_range in windows:
            run_id = f"walkforward_{test_range[0].date()}_{test_range[1].date()}"

            config = deepcopy(self.base_config)
            config.update(
                {
                    "start_date": str(test_range[0].date()),
                    "end_date": str(test_range[1].date()),
                    "run_id": run_id,
                    "output_dir": f"backtests/{run_id}",
                }
            )

            config_path = self.output_dir / f"{run_id}.yaml"
            save_yaml(config, config_path)

            print(f"▶️ Running walkforward test: {run_id}")
            run_backtest(config_path=str(config_path), use_cached_features=False)

            try:
                metrics = load_metrics_for_run(run_id)
            except FileNotFoundError:
                print(f"⚠️ metrics.json missing for run {run_id}, skipping.")
                continue

            results.append(
                {
                    "run_id": run_id,
                    "test_start": str(test_range[0].date()),
                    "test_end": str(test_range[1].date()),
                    "metrics": metrics,
                }
            )

        return sorted(results, key=lambda r: r["test_start"])
