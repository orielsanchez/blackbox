from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

from blackbox.cli_scripts.main import run_backtest
from blackbox.research.metrics import load_metrics_for_run
from blackbox.tuning.grid_utils import patch_config_with_metadata
from blackbox.tuning.interfaces import Tuner


class GridTuner(Tuner):
    def tune(
        self,
        config_path: Union[str, Path],
        metric: str,
        results_dir: Union[str, Path],
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.search_space:
            raise ValueError("Search space is empty")

        keys, values = zip(*self.search_space.items(), strict=True)
        combinations = list(product(*values))
        results = []

        for combo in combinations:
            param_dict = dict(zip(keys, combo, strict=True))
            run_id = "tune_" + "_".join(f"{k}-{v}" for k, v in param_dict.items())
            tmp_config_path = Path("/tmp") / f"{run_id}.yaml"

            patch_config_with_metadata(
                base_path=config_path,
                param_overrides=param_dict,
                run_id=run_id,
                out_path=tmp_config_path,
            )

            print(f"\n🚀 Running {run_id}")
            run_backtest(
                config_path=str(tmp_config_path),
                use_cached_features=False,
                refresh_data=False,
                plot_equity=False,
                output_dir=Path("backtests") / run_id,
            )

            try:
                metrics = load_metrics_for_run(run_id)
                score = self._extract_metric(metrics, metric)
                results.append((param_dict, score))
            except Exception as e:
                print(f"⚠️ Failed to load metrics for {run_id}: {e}")
                results.append((param_dict, float("-inf")))

        # Sort results by descending metric score
        results.sort(key=lambda x: x[1], reverse=True)

        # Save full results table
        df = pd.DataFrame([{**params, metric: score} for params, score in results])
        df.sort_values(metric, ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_dir / "results.csv", index=False)
        df.to_json(results_dir / "results.json", orient="records", indent=2)

        # Plot tuning curve
        plt.figure(figsize=(10, 4))
        df[metric].plot(marker="o")
        plt.title(f"Tuning Results ({metric})")
        plt.xlabel("Trial #")
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / f"{metric}_plot.png")
        plt.close()

        print("\n🏁 Top Grid Search Results:")
        print(df.head(5).to_markdown(index=False))

        return [
            ({key: row[key] for key in self.search_space}, row[metric])
            for _, row in df.iterrows()
        ]

    def _extract_metric(self, metrics: Dict[str, Any], metric: str) -> float:
        """
        Robustly extract a metric value by matching on normalized keys.

        Examples:
            'sharpe_ratio' matches 'Sharpe Ratio'
            'r_squared'    matches 'R squared'
        """
        normalized_target = metric.strip().lower().replace(" ", "_")
        for k, v in metrics.items():
            if k.strip().lower().replace(" ", "_") == normalized_target:
                return float(v)
        return float("-inf")
