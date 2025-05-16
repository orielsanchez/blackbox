from abc import ABC

import pandas as pd

from blackbox.feature_generators.resolve import (
    resolve_and_generate_features,
    resolve_feature_names,
)
from blackbox.models.interfaces import AlphaModel
from blackbox.utils.context import get_feature_matrix, get_logger


class FeatureAwareAlphaModel(AlphaModel, ABC):
    """
    Base class for alpha models that rely on a dynamic feature specification.
    Automatically handles resolution of missing features at runtime.
    """

    def __init__(self, features: list[dict]):
        self.logger = get_logger()
        self.feature_config = features or []
        self.feature_columns = resolve_feature_names(self.feature_config)
        self.logger.info(f"Resolved feature columns: {self.feature_columns}")

    def predict(self, snapshot: dict) -> pd.Series:
        """
        ML-compatible alias for generate method.
        By default, calls the concrete class's generate method.
        """
        return self.generate(snapshot)

    def get_feature_matrix_for(self, snapshot: dict) -> pd.DataFrame:
        """
        Extracts feature rows for the given snapshot date and resolves missing features if needed.
        """
        # Prefer feature_vector from snapshot if present (backtest pipeline)
        if "feature_vector" in snapshot and snapshot["feature_vector"] is not None:
            return snapshot["feature_vector"]
        # Fallback: use global context (for research/interactive use)
        try:
            feature_matrix = get_feature_matrix()
        except Exception:
            feature_matrix = None
        if feature_matrix is not None:
            ohlcv = snapshot.get("ohlcv")
            resolved = resolve_and_generate_features(
                self.feature_config,
                existing_matrix=feature_matrix,
                ohlcv=ohlcv,
            )
            return resolved
        raise RuntimeError("Feature matrix not set in snapshot or global context.")
