from typing import Dict, List, Optional

import pandas as pd

from blackbox.models.interfaces import AlphaModel
from blackbox.utils.context import get_logger
from blackbox.utils.signals import normalize_signal


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
        return "mean_reversion2"

    def log_info(self, msg: str):
        if self.verbose:
            self.logger.info(msg)

    def log_debug(self, msg: str):
        if self.verbose:
            self.logger.debug(msg)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("Expected MultiIndex [date, symbol], got single index")

        if features.index.names != ["date", "symbol"]:
            raise ValueError(
                f"Expected MultiIndex with levels ['date', 'symbol'], got {features.index.names}"
            )

        if not features.index.is_monotonic_increasing:
            self.logger.warning(
                "⚠️ Feature index not sorted, sorting now for multi-index slicing"
            )
            features = features.sort_index()

        dates = features.index.get_level_values("date").unique()
        if len(dates) != 1:
            raise ValueError(f"Expected single date, got {dates.tolist()}")
        date = dates[0]

        self.log_info(f"📅 Processing alpha for {date}")

        all_symbols = features.index.get_level_values("symbol").unique()
        symbols = (
            all_symbols.intersection(self.universe) if self.universe else all_symbols
        )

        if features.empty or symbols.empty:
            self.logger.warning(
                f"{date} | ⚠️ No symbols or features available after filtering"
            )
            return pd.Series(0.0, index=all_symbols, name="mean_reversion_signal")

        # ✅ Proper multi-index slicing
        features_today = features.loc[pd.IndexSlice[date, symbols], :]

        # Extract candidate z-score columns
        candidate_cols = []
        for spec in self.features:
            if "zscore" not in spec["name"]:
                continue
            period = spec["params"].get("period", 20)
            col_name = f"{spec['name']}_{period}"
            if col_name in features_today.columns:
                candidate_cols.append(col_name)

        if not candidate_cols:
            self.logger.warning(f"{date} | ⚠️ No z-score features found in columns")
            return pd.Series(0.0, index=symbols, name="mean_reversion_signal")

        feature_col = candidate_cols[0]
        raw_scores = -features_today[feature_col]  # Invert for mean reversion
        raw_scores = raw_scores.dropna()

        self.log_debug(f"{date} | Raw signals: {raw_scores.to_dict()}")

        # Apply threshold filter
        filtered = raw_scores[raw_scores.abs() >= self.threshold]
        self.log_debug(
            f"{date} | Signals after threshold ({self.threshold}): {len(filtered)}"
        )

        if not filtered.empty and self.verbose:
            buys = filtered.sort_values().head(3)
            sells = filtered.sort_values(ascending=False).head(3)
            self.logger.info(f"{date} | Top buys: {buys.to_dict()}")
            self.logger.info(f"{date} | Top sells: {sells.to_dict()}")

        # Return full signal vector: 0.0 for others

        # Return full signal vector: 0.0 for others
        full_index = pd.MultiIndex.from_product(
            [[date], symbols], names=["date", "symbol"]
        )
        signals = pd.Series(0.0, index=full_index, name="mean_reversion_signal")
        signals.loc[filtered.index] = filtered

        return normalize_signal(signals)
