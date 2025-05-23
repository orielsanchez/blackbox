import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from blackbox.cli_scripts.main import run_backtest
from blackbox.research.metrics import load_metrics_for_run
from blackbox.utils.io import load_yaml, save_yaml


def set_nested(config: Dict[str, Any], path: List[Any], value: Any) -> None:
    """Set a value in a nested dictionary via a list path."""
    for key in path[:-1]:
        config = config[key]
    config[path[-1]] = value


def _find_feature_index(features: list[dict], feature_name: str) -> int:
    """Find the index of a named feature in a submodel's features list."""
    available = [f.get("name") for f in features]
    for i, f in enumerate(features):
        if f.get("name") == feature_name:
            return i
    raise ValueError(
        f"\n❌ Feature '{feature_name}' not found in features list.\n"
        f"🔍 Available features: {available}"
    )


def resolve_param_path(config: dict, key: str) -> list:
    """
    Supports nested submodel keys using double-underscore syntax.
    Example:
        mean_reversion__rsi_period → alpha_model.params.mean_reversion_model.params.features[:].rsi.params.period
    """
    alpha_model = config.get("alpha_model", {})
    alpha_params = alpha_model.get("params", {})

    if "__" not in key:
        raise KeyError(f"No resolver defined for tuning key: '{key}'")

    submodel_key, feature_key = key.split("__", 1)
    submodel_name = f"{submodel_key}_model"

    submodel = alpha_params.get(submodel_name, {})
    submodel_params = submodel.get("params", {})
    features = submodel_params.get("features", [])

    if feature_key == "threshold":
        return [
            "alpha_model",
            "params",
            submodel_name,
            "params",
            "threshold",
        ]
    elif feature_key == "boll_stddev":
        idx = _find_feature_index(features, "bollinger_band")
        return [
            "alpha_model",
            "params",
            submodel_name,
            "params",
            "features",
            idx,
            "params",
            "std_dev",
        ]
    elif feature_key == "rsi_period":
        idx = _find_feature_index(features, "rsi")
        return [
            "alpha_model",
            "params",
            submodel_name,
            "params",
            "features",
            idx,
            "params",
            "period",
        ]
    elif feature_key == "zscore_period":
        idx = _find_feature_index(features, "rolling_zscore")
        return [
            "alpha_model",
            "params",
            submodel_name,
            "params",
            "features",
            idx,
            "params",
            "period",
        ]

    elif feature_key == "vol_span":
        idx = _find_feature_index(features, "ewma_volatility")
        return [
            "alpha_model",
            "params",
            submodel_name,
            "params",
            "features",
            idx,
            "params",
            "span",
        ]

    raise KeyError(f"Unsupported nested tuning key: '{key}'")


def update_yaml_config(
    base_path: Union[str, Path],
    param_overrides: Dict[str, Any],
    out_path: Union[str, Path],
) -> None:
    config = load_yaml(base_path)

    for key, value in param_overrides.items():
        try:
            path = resolve_param_path(config, key)
            print(f"✅ Setting {key} = {value} at: {'.'.join(map(str, path))}")
            set_nested(config, path, value)
        except Exception as e:
            raise RuntimeError(f"Failed to update param '{key}': {e}")

    save_yaml(config, out_path)


def run_grid_search(
    config_path: Union[str, Path],
    search_space: Dict[str, List[Any]],
    metric: str = "sharpe",
    results_dir: Union[str, Path] = "grid_search_results",
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Run a grid search over parameter combinations and evaluate backtest performance.

    Args:
        config_path: Base YAML config file to start from.
        search_space: Dict of param_name → list of values to try.
        metric: Metric to optimize (e.g., "sharpe", "return", etc.).
        results_dir: Where to save the CSV/JSON output summary.

    Returns:
        Sorted list of (param_dict, score) tuples from best to worst.
    """
    if not search_space:
        raise ValueError("Search space cannot be empty.")

    keys, values = zip(*search_space.items(), strict=True)
    combinations = list(product(*values))
    results = []

    for combo in combinations:
        param_dict = dict(zip(keys, combo, strict=True))
        run_id = "tune_" + "_".join(f"{k}-{v}" for k, v in param_dict.items())
        tmp_config_path = f"/tmp/{run_id}.yaml"

        update_yaml_config(config_path, param_dict, tmp_config_path)

        print(f"\n🔧 Running {run_id}")
        run_backtest(
            config_path=tmp_config_path,
            use_cached_features=False,
            refresh_data=False,
            plot_equity=False,
            output_dir=Path("backtests") / run_id,
        )

        try:
            metrics = load_metrics_for_run(run_id)
            score = metrics.get(metric, float("-inf"))
            results.append((param_dict, score))
        except Exception as e:
            print(f"⚠️ Failed to load metrics for {run_id}: {e}")
            results.append((param_dict, float("-inf")))

    # Sort descending by selected metric
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n🏁 Top Results:")
    for params, score in results[:5]:
        print(f"{params} → {metric}: {score:.4f}")

    # ────── Save to CSV/JSON ──────
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([{**params, metric: score} for params, score in results])
    df.to_csv(results_path / "grid_search_results.csv", index=False)

    with open(results_path / "grid_search_results.json", "w") as f:
        json.dump([{"params": p, metric: s} for p, s in results], f, indent=2)

    print(f"\n✅ Saved results to {results_path}/grid_search_results.csv and .json")

    return results
