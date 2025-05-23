# src/blackbox/tuning/tuned_walkforward_runner.py
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from blackbox.core.runner import run_backtest
from blackbox.research.metrics import load_metrics_for_run
from blackbox.tuning.grid_utils import patch_config_with_metadata
from blackbox.tuning.interfaces import Tuner
from blackbox.tuning.walkforward import generate_walkforward_windows
from blackbox.utils.io import load_yaml, save_yaml


class TunedWalkforwardRunner:
    """
    1. Tunes hyper-parameters on a *train* window with the supplied `Tuner`.
    2. Applies the best params to the *test* window.
    3. Rolls end-equity forward so account capital compounds across windows.
    """

    def __init__(
        self,
        base_config_path: str | Path,
        tuner: Tuner,
        train_days: int,
        test_days: int,
        output_dir: str | Path = "tmp/tuned_walkforward_configs",
        metric: str = "Sharpe Ratio",
    ) -> None:
        self.base_config_path = Path(base_config_path)
        self.tuner = tuner
        self.train_days = train_days
        self.test_days = test_days
        self.output_dir = Path(output_dir)
        self.metric = metric

        self.base_config: Dict[str, Any] = load_yaml(self.base_config_path)
        self.start_date: str = self.base_config["start_date"]
        self.end_date: str = self.base_config["end_date"]
        self.initial_capital: float = float(
            self.base_config.get("initial_portfolio_value", 10_000)
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def run(self) -> List[Dict[str, Any]]:
        """
        Execute the entire walk-forward procedure and return per-window results.
        Capital is compounded by rolling the end equity into the next window.
        """
        results: List[Dict[str, Any]] = []
        windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = generate_walkforward_windows(
            self.start_date, self.end_date, self.train_days, self.test_days
        )

        capital: float = self.initial_capital

        for train_range, test_range in windows:
            train_id = f"train_{train_range[0].date()}_{train_range[1].date()}"
            test_id = f"test_{test_range[0].date()}_{test_range[1].date()}"

            # ── 1. prepare training config ──────────────────────────────
            train_cfg: Dict[str, Any] = deepcopy(self.base_config)
            train_cfg.update(
                {
                    "start_date": str(train_range[0].date()),
                    "end_date": str(train_range[1].date()),
                    "run_id": train_id,
                    "output_dir": str(self.output_dir / train_id),
                    "initial_portfolio_value": capital,
                }
            )
            train_cfg_path = self.output_dir / f"{train_id}.yaml"
            save_yaml(train_cfg, train_cfg_path)

            # ── 2. hyper-parameter tuning ───────────────────────────────
            print(f"\n🔍 Tuning on train window: {train_id}")
            best_trials = self.tuner.tune(
                config_path=train_cfg_path,
                metric=self.metric,
                results_dir=self.output_dir / f"{train_id}_search",
            )

            if not best_trials:
                print(f"⚠️ No tuning results for {train_id} — skipping.")
                continue

            best_params, best_score = best_trials[0]
            print(f"✅ Best params: {best_params} | {self.metric}: {best_score:.4f}")

            # ── 3. build test config (two-step patch) ───────────────────
            # 3-a: write YAML with tuned params only
            test_cfg_path = self.output_dir / f"{test_id}.yaml"
            patch_config_with_metadata(
                base_path=self.base_config_path,
                param_overrides=best_params,
                run_id=test_id,
                out_path=test_cfg_path,
            )
            # 3-b: inject capital + date range directly
            test_cfg = load_yaml(test_cfg_path)
            test_cfg.update(
                {
                    "start_date": str(test_range[0].date()),
                    "end_date": str(test_range[1].date()),
                    "initial_portfolio_value": capital,
                    "output_dir": str(self.output_dir / test_id),
                }
            )
            save_yaml(test_cfg, test_cfg_path)

            # ── 4. run test back-test ───────────────────────────────────
            print(f"🏁 Running backtest: {test_id}")
            run_backtest(
                config_path=str(test_cfg_path),
                use_cached_features=False,
                output_dir=self.output_dir / test_id,
            )

            # ── 5. gather metrics & roll capital ───────────────────────────────
            try:
                metrics_path = self.output_dir / test_id / "metrics.json"
                with open(metrics_path) as fh:
                    raw_json = json.load(fh)

                # the file structure is {"metrics": {...}, "metadata": {...}}
                metrics = raw_json["metrics"]
                window_score = self._extract_metric(metrics, self.metric)
                capital = self._extract_end_equity(test_id)  # ← no metrics arg
            except Exception as e:
                print(f"⚠️ Could not process metrics for {test_id}: {e}")
                continue

            results.append(
                {
                    "run_id": test_id,
                    "test_start": str(test_range[0].date()),
                    "test_end": str(test_range[1].date()),
                    "params": best_params,
                    "metrics": metrics,
                    "score": window_score,
                    "end_equity": capital,
                }
            )

        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_metric(metrics: Dict[str, Any], metric: str) -> float:
        tgt = metric.strip().lower().replace(" ", "_")
        for k, v in metrics.items():
            if k.strip().lower().replace(" ", "_") == tgt:
                return float(v)
        raise KeyError(f"Metric '{metric}' not found in {list(metrics.keys())}")

    def _extract_end_equity(self, run_id: str) -> float:
        """
        Read the last equity value from the run's equity_curve.csv, which now
        lives under self.output_dir/<run_id>/.
        """
        curve_path = self.output_dir / run_id / "equity_curve.csv"
        if curve_path.exists():
            curve = pd.read_csv(curve_path)
            if not curve.empty:
                return float(curve.iloc[-1, 1])  # assumes [date, equity]
        raise FileNotFoundError(f"{curve_path} missing or empty")
