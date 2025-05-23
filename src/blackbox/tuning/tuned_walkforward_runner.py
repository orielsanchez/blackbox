from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from blackbox.core.runner import run_backtest
from blackbox.research.metrics import load_metrics_for_run
from blackbox.tuning.grid_utils import patch_config_with_metadata
from blackbox.tuning.interfaces import Tuner
from blackbox.tuning.walkforward import generate_walkforward_windows
from blackbox.utils.io import load_yaml, save_yaml


class TunedWalkforwardRunner:
    def __init__(
        self,
        base_config_path: str,
        tuner: Tuner,
        train_days: int,
        test_days: int,
        output_dir: str = "tmp/tuned_walkforward_configs",
        metric: str = "Sharpe Ratio",  # human-readable default
    ):
        self.base_config_path = Path(base_config_path)
        self.tuner = tuner
        self.train_days = train_days
        self.test_days = test_days
        self.output_dir = Path(output_dir)
        self.metric = metric
        self.base_config = load_yaml(self.base_config_path)
        self.start_date = self.base_config["start_date"]
        self.end_date = self.base_config["end_date"]

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[Dict[str, Any]]:
        results = []
        windows = generate_walkforward_windows(
            self.start_date, self.end_date, self.train_days, self.test_days
        )

        for train_range, test_range in windows:
            train_id = f"train_{train_range[0].date()}_{train_range[1].date()}"
            test_id = f"test_{test_range[0].date()}_{test_range[1].date()}"

            # ─── Generate training config ───
            train_config = deepcopy(self.base_config)
            train_config.update(
                {
                    "start_date": str(train_range[0].date()),
                    "end_date": str(train_range[1].date()),
                    "run_id": train_id,
                    "output_dir": f"backtests/{train_id}",
                }
            )
            train_config_path = self.output_dir / f"{train_id}.yaml"
            save_yaml(train_config, train_config_path)

            # ─── Grid search ───
            print(f"\n🔍 Tuning on train window: {train_id}")
            best_results = self.tuner.tune(
                config_path=train_config_path,
                metric=self.metric,
                results_dir=self.output_dir / f"{train_id}_search",
            )

            if not best_results:
                print(f"⚠️ No tuning results for {train_id}, skipping.")
                continue

            best_params, best_score = best_results[0]
            print(f"✅ Best: {best_params} | {self.metric}: {best_score:.4f}")

            # ─── Patch and run test window ───
            test_config_path = self.output_dir / f"{test_id}.yaml"
            patch_config_with_metadata(
                base_path=self.base_config_path,
                param_overrides=best_params,
                run_id=test_id,
                out_path=test_config_path,
            )

            print(f"🏁 Running backtest: {test_id}")
            run_backtest(
                config_path=str(test_config_path),
                use_cached_features=False,
            )

            # ─── Load metrics ───
            try:
                metrics_dict = load_metrics_for_run(test_id)
                if "metrics" in metrics_dict:
                    metrics = metrics_dict["metrics"]
                else:
                    metrics = metrics_dict
                extracted_score = self._extract_metric(metrics, self.metric)
            except Exception as e:
                print(f"⚠️ Failed to load or extract metrics for {test_id}: {e}")
                continue

            results.append(
                {
                    "run_id": test_id,
                    "test_start": str(test_range[0].date()),
                    "test_end": str(test_range[1].date()),
                    "params": best_params,
                    "metrics": metrics,
                    "score": extracted_score,
                }
            )

        return results

    def _extract_metric(self, metrics: Dict[str, Any], metric: str) -> float:
        """Match metric key regardless of formatting differences."""
        target = metric.strip().lower().replace(" ", "_")
        for k, v in metrics.items():
            normalized_key = k.strip().lower().replace(" ", "_")
            if normalized_key == target:
                return float(v)
        raise KeyError(f"Metric '{metric}' not found in: {list(metrics.keys())}")
