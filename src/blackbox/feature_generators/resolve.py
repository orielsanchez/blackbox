# src/blackbox/feature_generators/resolve.py

import pandas as pd

from blackbox.feature_generators.base import feature_registry
from blackbox.utils.context import get_logger


def resolve_feature_columns(features: list[dict]) -> dict[str, str]:
    """
    Given a list of feature configs, return a mapping from logical feature types
    (e.g. 'zscore', 'bollinger') to resolved DataFrame column names.
    """
    columns = {}
    for feature in features or []:
        name = feature["name"]
        params = feature.get("params", {})
        period = params.get("period")

        if name == "zscore_price":
            columns["zscore"] = f"{name}_{period}"
        elif name == "bollinger_band":
            columns["bollinger"] = f"bollinger_norm_{period}"
        else:
            columns[name] = f"{name}_{period}" if period else name

    return columns


def resolve_and_generate_features(
    features: list[dict],
    feature_matrix: pd.DataFrame,
    ohlcv: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure all required feature columns exist in the feature matrix.
    Generate them using registered feature classes if missing.
    """
    for feature in features or []:
        name = feature["name"]
        params = feature.get("params", {})
        period = params.get("period")

        if name == "zscore_price":
            col = f"{name}_{period}"
        elif name == "bollinger_band":
            col = f"bollinger_norm_{period}"
        else:
            col = f"{name}_{period}" if period else name

        if col in feature_matrix.columns:
            continue  # already present

        if name not in feature_registry:
            raise ValueError(f"❌ Unknown feature: {name}")

        logger = get_logger()
        logger.debug(f"🛠 Generating missing feature: {col}")

        generator_cls = feature_registry[name]
        generator = generator_cls(**params)
        output = generator.run(ohlcv)

        if isinstance(output, pd.Series):
            feature_matrix[col] = output
        elif isinstance(output, pd.DataFrame):
            for c in output.columns:
                feature_matrix[c] = output[c]
        else:
            raise TypeError(
                f"❌ Feature {name} returned unsupported type: {type(output)}"
            )

    return feature_matrix
