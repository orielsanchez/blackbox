from typing import Dict, List, Optional

import pandas as pd

from blackbox.models.interfaces import AlphaModel
from blackbox.utils.context import get_logger
from blackbox.utils.signals import normalize_signal


class MeanReversionAlphaModel(AlphaModel):
    """Mean reversion alpha model using composite features filtered by RSI."""

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

    def predict(self, features: pd.DataFrame) -> pd.Series:
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
        if self.verbose:
            self.logger.info(f"📅 Processing alpha for {date}")

        all_symbols = features.index.get_level_values("symbol").unique()
        symbols = (
            all_symbols.intersection(self.universe) if self.universe else all_symbols
        )
        full_index = pd.MultiIndex.from_product(
            [[date], symbols], names=["date", "symbol"]
        )

        if features.empty or symbols.empty:
            self.logger.warning(f"{date} | ⚠️ No symbols or features after filtering")
            return pd.Series(0.0, index=full_index, name="mean_reversion_signal")

        features_today = features.loc[pd.IndexSlice[date, symbols], :]

        # Extract feature columns
        rsi_col = None
        signal_parts = []

        for f in self.features:
            name = f["name"]
            params = f.get("params", {})
            period = params.get("period", 14 if name == "rsi" else 20)
            col = f"{name}_{period}"

            if name == "rsi":
                if col in features_today:
                    rsi_col = col
            else:
                if col in features_today:
                    # Invert zscore-type signals for mean-reversion
                    if "zscore" in name:
                        signal_parts.append(-features_today[col])
                    else:
                        signal_parts.append(features_today[col])

        if not rsi_col or not signal_parts:
            self.logger.warning(f"{date} | ⚠️ Missing RSI or signal features")
            return pd.Series(0.0, index=full_index, name="mean_reversion_signal")

        # Compute composite raw score
        raw_score = sum(signal_parts) / len(signal_parts)
        rsi = features_today[rsi_col]
        mask = (rsi < 30) | (rsi > 70)

        filtered = raw_score.where(mask, 0.0).dropna()

        # Normalize + top-N filter
        normalized = normalize_signal(filtered)
        top_n = 50
        normalized = normalized.loc[
            normalized.abs().sort_values(ascending=False).head(top_n).index
        ]

        # Apply threshold
        self.logger.info(
            f"{date} | Signal stats: mean={normalized.mean():.3f}, std={normalized.std():.3f}, max={normalized.max():.3f}"
        )
        strong_signals = normalized[normalized.abs() >= self.threshold]

        if self.verbose:
            self.logger.debug(
                f"{date} | {len(strong_signals)} signals passed threshold"
            )
            if not strong_signals.empty:
                buys = strong_signals.sort_values().head(3)
                sells = strong_signals.sort_values(ascending=False).head(3)
                self.logger.info(f"{date} | Top buys: {buys.to_dict()}")
                self.logger.info(f"{date} | Top sells: {sells.to_dict()}")

        # Output full signal vector
        signals = pd.Series(0.0, index=full_index, name="mean_reversion_signal")
        signals.loc[strong_signals.index] = strong_signals
        return signals
