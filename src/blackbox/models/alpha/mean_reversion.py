from typing import Dict, List, Optional

import pandas as pd

from blackbox.models.interfaces import AlphaModel
from blackbox.utils.context import get_logger


class MeanReversionAlphaModel(AlphaModel):
    """Mean reversion alpha model using z-score features."""

    def __init__(
        self,
        threshold: float = 0.5,
        features: Optional[List[Dict]] = None,
        universe: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.threshold = threshold
        self.features = features or []
        self.universe = universe or []
        self.verbose = verbose
        self.logger = get_logger()

    @property
    def name(self) -> str:
        return "mean_reversion"

    def log_info(self, msg: str):
        if self.verbose:
            self.logger.info(msg)

    def log_debug(self, msg: str):
        if self.verbose:
            self.logger.debug(msg)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("Expected MultiIndex [date, symbol]")

        dates = features.index.get_level_values("date").unique()
        if len(dates) != 1:
            raise ValueError(f"Expected single date, got {dates.tolist()}")
        date = dates[0]

        self.log_info(f"📅 Processing alpha for {date}")

        symbols = features.index.get_level_values("symbol")
        if self.universe:
            symbols = symbols.intersection(self.universe)
            features_today = features.loc[(date, symbols)]

        if features.empty:
            self.logger.warning(f"{date} | ⚠️ No features available after filtering")
            return pd.Series(dtype=float)

        candidate_cols = []
        for spec in self.features:
            if "zscore" not in spec["name"]:
                continue
            period = spec["params"].get("period", 20)
            col_name = f"{spec['name']}_{period}"
            if col_name in features.columns:
                candidate_cols.append(col_name)

        if not candidate_cols:
            self.logger.warning(f"{date} | ⚠️ No z-score features found in columns")
            return pd.Series(dtype=float)

        feature_col = candidate_cols[0]
        signals = -features[feature_col]
        signals = signals.dropna()

        self.log_debug(f"{date} | Raw signals: {signals.to_dict()}")

        filtered = signals[signals.abs() >= self.threshold]
        self.log_debug(
            f"{date} | Signals after threshold ({self.threshold}): {len(filtered)}"
        )

        if not filtered.empty and self.verbose:
            buys = filtered.sort_values().head(3)
            sells = filtered.sort_values(ascending=False).head(3)
            self.logger.info(f"{date} | Top buys: {buys.to_dict()}")
            self.logger.info(f"{date} | Top sells: {sells.to_dict()}")

        return filtered
