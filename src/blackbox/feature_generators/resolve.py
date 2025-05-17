# src/blackbox/feature_generators/resolve.py

from typing import Any

import pandas as pd

from blackbox.feature_generators.base import feature_registry
from blackbox.utils.context import get_logger


def resolve_feature_names(features: list[dict[str, Any]]) -> dict[str, str]:
    """
    Map logical feature types to actual output column names in the feature matrix.
    """
    mapping = {}
    for feature in features or []:
        name = feature["name"]
        params = feature.get("params", {})
        period = params.get("period")

        if name == "bollinger_band":
            col = f"bollinger_norm_{period}" if period else "bollinger_norm"
        elif name == "zscore_price":
            col = f"zscore_price_{period}" if period else "zscore_price"
        else:
            col = f"{name}_{period}" if period else name

        mapping[name] = col

    return mapping


def resolve_and_generate_features(
    features: list[dict[str, Any]],
    existing_matrix: pd.DataFrame,
    ohlcv: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensures all required features are present in the matrix. If not, generates missing ones.
    Returns a new DataFrame with the full feature set.
    """
    logger = get_logger()
    new_features = []

    for feature in features or []:
        name = feature["name"]
        params = feature.get("params", {})

        generator_cls = feature_registry.get(name)
        if generator_cls is None:
            raise ValueError(f"❌ Unknown feature: {name}")

        generator = generator_cls(**params)
        output = generator.generate(ohlcv)  # ✅ FIXED: use generate()

        for col_name in output.columns:
            if col_name in existing_matrix.columns:
                continue
            logger.debug(f"🛠 Generating missing feature: {col_name}")
            new_features.append(output[[col_name]])

    if new_features:
        return pd.concat([existing_matrix] + new_features, axis=1)

    return existing_matrix
