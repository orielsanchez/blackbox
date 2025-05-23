from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from blackbox.utils.io import load_yaml, save_yaml


def to_builtin_scalar(x: Any) -> Any:
    """Convert NumPy scalars (e.g. np.float64) to native Python types."""
    if isinstance(x, np.generic):
        return x.item()
    return x


def set_nested(config: Dict[str, Any], path: List[Any], value: Any) -> None:
    """Set a value deep in a nested dict given a path of keys."""
    for key in path[:-1]:
        config = config[key]
    config[path[-1]] = value


def find_feature_index(features: List[dict], name: str) -> int:
    for i, f in enumerate(features):
        if f.get("name") == name:
            return i
    raise ValueError(
        f"Feature '{name}' not found in: {[f.get('name') for f in features]}"
    )


def resolve_param_path(config: dict, key: str) -> List[Any]:
    """
    Map tuning keys like 'mean_reversion__rsi_period' to nested YAML paths.
    """
    if "__" not in key:
        raise KeyError(f"No resolver defined for key: '{key}'")

    model_key, param_key = key.split("__", 1)
    model_name = f"{model_key}_model"

    features = (
        config.get("alpha_model", {})
        .get("params", {})
        .get(model_name, {})
        .get("params", {})
        .get("features", [])
    )

    if param_key == "threshold":
        return ["alpha_model", "params", model_name, "params", "threshold"]
    elif param_key == "boll_stddev":
        idx = find_feature_index(features, "bollinger_band")
        return [
            "alpha_model",
            "params",
            model_name,
            "params",
            "features",
            idx,
            "params",
            "std_dev",
        ]
    elif param_key == "rsi_period":
        idx = find_feature_index(features, "rsi")
        return [
            "alpha_model",
            "params",
            model_name,
            "params",
            "features",
            idx,
            "params",
            "period",
        ]
    elif param_key == "zscore_period":
        idx = find_feature_index(features, "rolling_zscore")
        return [
            "alpha_model",
            "params",
            model_name,
            "params",
            "features",
            idx,
            "params",
            "period",
        ]
    elif param_key == "vol_span":
        idx = find_feature_index(features, "ewma_volatility")
        return [
            "alpha_model",
            "params",
            model_name,
            "params",
            "features",
            idx,
            "params",
            "span",
        ]

    raise KeyError(f"Unsupported tuning key: '{key}'")


def patch_config_with_metadata(
    base_path: Union[str, Path],
    param_overrides: Dict[str, Any],
    run_id: str,
    out_path: Union[str, Path],
    output_root: Union[str, Path] = "backtests",
) -> None:
    """
    Load base config, apply parameter overrides and metadata, and save the result.
    """
    config = load_yaml(base_path)

    for key, value in param_overrides.items():
        try:
            path = resolve_param_path(config, key)
            set_nested(config, path, to_builtin_scalar(value))
        except Exception as e:
            raise RuntimeError(f"Failed to patch '{key}': {e}")

    config["run_id"] = run_id
    config["output_dir"] = str(Path(output_root) / run_id)
    save_yaml(config, out_path)
