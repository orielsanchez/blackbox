from typing import Dict, List, Optional

import pandas as pd

from blackbox.feature_generators.pipeline import FeaturePipeline
from blackbox.models.interfaces import AlphaModel
from blackbox.utils.context import get_logger
from blackbox.utils.signals import normalize_signal


class MomentumAlphaModel(AlphaModel):
    def __init__(
        self,
        short_momentum: int = 5,
        long_momentum: int = 20,
        ema_short: int = 10,
        ema_long: int = 50,
        signal_weights: Optional[Dict[str, float]] = None,
        min_signal_threshold: float = 0.01,
        universe: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.short_momentum = short_momentum
        self.long_momentum = long_momentum
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.min_signal_threshold = min_signal_threshold
        self.universe = universe or []
        self.verbose = verbose
        self.logger = get_logger()

        self.signal_weights = signal_weights or {
            "momentum_short": 0.4,
            "momentum_long": 0.4,
            "ema_diff": 0.2,
        }

        self.feature_pipeline = FeaturePipeline(
            [
                {"name": "momentum", "params": {"period": self.short_momentum}},
                {"name": "momentum", "params": {"period": self.long_momentum}},
                {
                    "name": "ema_crossover",
                    "params": {"short": self.ema_short, "long": self.ema_long},
                },
            ]
        )

    @property
    def name(self) -> str:
        return "momentum_alpha"

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Args:
            features: pd.DataFrame with MultiIndex [date, symbol]
        Returns:
            pd.Series [symbol → signal] for a single date
        """
        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("Expected MultiIndex [date, symbol]")

        if features.index.names != ["date", "symbol"]:
            raise ValueError(
                f"Expected index names ['date', 'symbol'], got {features.index.names}"
            )

        if not features.index.is_monotonic_increasing:
            self.logger.warning("⚠️ Feature index not sorted. Sorting now.")
            features = features.sort_index()

        date = features.index.get_level_values("date").unique()
        if len(date) != 1:
            raise ValueError(f"Expected single date, got {date.tolist()}")
        date = date[0]

        features_today = features.loc[pd.IndexSlice[date, :], :]

        if self.universe:
            symbols = features_today.index.get_level_values("symbol").unique()
            filtered_symbols = symbols.intersection(self.universe)
            features_today = features_today.loc[pd.IndexSlice[:, filtered_symbols], :]

        if features_today.empty:
            self.logger.warning(f"{date} | ⚠️ No features for universe")
            return pd.Series(0.0, index=features_today.index, name="momentum_signal")

        col_short = f"momentum_{self.short_momentum}"
        col_long = f"momentum_{self.long_momentum}"
        col_ema_diff = f"ema_{self.ema_short}_{self.ema_long}_diff"

        for col in [col_short, col_long, col_ema_diff]:
            if col not in features_today.columns:
                self.logger.warning(f"{date} | ⚠️ Missing expected feature: {col}")
                return pd.Series(dtype=float)

        score = (
            self.signal_weights["momentum_short"] * features_today[col_short]
            + self.signal_weights["momentum_long"] * features_today[col_long]
            + self.signal_weights["ema_diff"] * features_today[col_ema_diff]
        )

        score = score.where(score.abs() > self.min_signal_threshold, 0.0)
        normalized = normalize_signal(score)

        if self.verbose:
            self.logger.info(
                f"{date} | Momentum signal stats: mean={normalized.mean():.3f}, std={normalized.std():.3f}"
            )

        return normalized
