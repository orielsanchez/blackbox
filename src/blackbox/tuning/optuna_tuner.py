from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import optuna
import pandas as pd

from blackbox.cli_scripts.main import run_backtest
from blackbox.research.metrics import load_metrics_for_run
from blackbox.tuning.grid_utils import patch_config_with_metadata
from blackbox.tuning.interfaces import Tuner


class OptunaTuner(Tuner):
    def __init__(
        self,
        param_space: Dict[str, Union[List[Any], Tuple[Any, Any]]],
        n_trials: int = 30,
        direction: str = "maximize",
        n_jobs: int = 1,
    ):
        """
        param_space:
          Dict of param_name -> list of discrete values OR tuple (min, max) for float range
        n_jobs:
          Number of parallel jobs (processes) to run (default 1)
        """
        self.param_space = param_space
        self.n_trials = n_trials
        self.direction = direction
        self.n_jobs = n_jobs

    def tune(
        self, config_path: Union[str, Path], metric: str, results_dir: Union[str, Path]
    ) -> List[Tuple[Dict[str, Any], float]]:
        run_id_prefix = Path(config_path).stem
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        def objective(trial: optuna.Trial) -> float:
            sampled_params = self._sample_from_space(trial)

            run_id = f"{run_id_prefix}_optuna_{trial.number}"
            trial_config_path = results_dir / f"{run_id}.yaml"

            patch_config_with_metadata(
                base_path=config_path,
                param_overrides=sampled_params,
                run_id=run_id,
                out_path=trial_config_path,
            )

            try:
                run_backtest(str(trial_config_path), use_cached_features=True)
                metrics = load_metrics_for_run(run_id)
                score = self._extract_metric(metrics, metric)
                return score
            except Exception as e:
                print(f"⚠️ Trial {trial.number} failed: {e}")
                return float("-inf") if self.direction == "maximize" else float("inf")

        study = optuna.create_study(
            direction=self.direction, study_name=f"tune_{run_id_prefix}"
        )
        study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs)

        # Collect and save all trial results
        records = []
        for trial in study.trials:
            records.append({**trial.params, metric: trial.value})
        df = (
            pd.DataFrame(records)
            .sort_values(metric, ascending=False)
            .reset_index(drop=True)
        )

        df.to_csv(results_dir / "results.csv", index=False)
        df.to_json(results_dir / "results.json", orient="records", indent=2)

        plt.figure(figsize=(10, 4))
        df[metric].plot(marker="o")
        plt.title(f"Tuning Results ({metric})")
        plt.xlabel("Trial #")
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / f"{metric}_plot.png")
        plt.close()

        print("\n🏁 Top Optuna Search Results:")
        print(df.head(5).to_markdown(index=False))

        return [(row.drop(metric).to_dict(), row[metric]) for _, row in df.iterrows()]

    def _sample_from_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        for name, values in self.param_space.items():
            if (
                isinstance(values, tuple)
                and len(values) == 2
                and all(isinstance(v, (int, float)) for v in values)
            ):
                params[name] = trial.suggest_float(name, values[0], values[1])
            elif isinstance(values, list):
                params[name] = trial.suggest_categorical(name, values)
            else:
                params[name] = trial.suggest_categorical(name, values)
        return params

    def _extract_metric(self, metrics: Dict[str, Any], metric: str) -> float:
        normalized_target = metric.strip().lower().replace(" ", "_")
        for k, v in metrics.get("metrics", metrics).items():
            if k.strip().lower().replace(" ", "_") == normalized_target:
                return float(v)
        return float("-inf")
