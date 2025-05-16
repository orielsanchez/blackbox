# src/blackbox/feature_generators/resolve.py

import pandas as pd

from blackbox.feature_generators.base import feature_registry
from blackbox.utils.context import get_logger


def resolve_feature_names(features: list[dict]) -> dict[str, str]:
    """
    Map logical feature types to column names used in final DataFrame.
    Handles special cases where the output column name differs from the feature name.
    """
    mapping = {}
    for feature in features or []:
        name = feature["name"]
        params = feature.get("params", {})
        period = params.get("period")
        if name == "bollinger_band":
            # Output column is 'bollinger_norm_{period}'
            col = f"bollinger_norm_{period}" if period else "bollinger_norm"
        elif name == "zscore_price":
            col = f"zscore_price_{period}" if period else "zscore_price"
        else:
            # Default: name or name_period
            col = f"{name}_{period}" if period else name
        mapping[name] = col
    return mapping


def resolve_and_generate_features(
    features: list[dict],
    existing_matrix: pd.DataFrame,
    ohlcv: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure all requested features exist in the feature matrix.
    Generate missing ones from registered generators.
    Returns a new DataFrame with the full set of features.
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
        output = generator.run(ohlcv)

        for col_name in output.columns:
            if col_name in existing_matrix.columns:
                continue  # Already exists
            logger.debug(f"🛠 Generating missing feature: {col_name}")
            new_features.append(output[[col_name]])

    if new_features:
        return pd.concat([existing_matrix] + new_features, axis=1)

    return existing_matrix
