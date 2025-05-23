from typing import Dict, List, Optional

import pandas as pd

from blackbox.models.interfaces import AlphaModel
from blackbox.utils.context import get_logger
from blackbox.utils.signals import normalize_signal


class MeanReversionAlphaModel(AlphaModel):
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
        if not isinstance(features.index, pd.MultiIndex) or features.index.names != [
            "date",
            "symbol",
        ]:
            raise ValueError("Expected MultiIndex with names ['date', 'symbol']")

        if not features.index.is_monotonic_increasing:
            self.logger.warning("⚠️ Feature index not sorted. Sorting now.")
            features = features.sort_index()

        dates = features.index.get_level_values("date").unique()
        if len(dates) != 1:
            raise ValueError(f"Expected a single date. Got: {dates.tolist()}")
        date = dates[0]

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

        # Dynamically select and compose signals
        rsi_col = None
        signal_parts = []

        for f in self.features:
            name = f["name"]
            params = f.get("params", {})
            period = params.get("period", 14 if name == "rsi" else 20)
            output = params.get("output")  # Optional override
            col = output or f"{name}_{period}"

            if col not in features_today:
                self.logger.debug(f"{date} | Skipping missing feature column: {col}")
                continue

            if name == "rsi":
                rsi_col = col
            else:
                signal_part = (
                    -features_today[col] if "zscore" in name else features_today[col]
                )
                signal_parts.append(signal_part)

        if not signal_parts:
            self.logger.warning(f"{date} | ❌ No usable signal features found.")
            return pd.Series(0.0, index=full_index, name="mean_reversion_signal")

        raw_score = sum(signal_parts) / len(signal_parts)

        if rsi_col and rsi_col in features_today:
            rsi = features_today[rsi_col]
            raw_score = raw_score.where((rsi < 30) | (rsi > 70), 0.0)

        raw_score = raw_score.dropna()

        if raw_score.empty or raw_score.std() == 0:
            self.logger.debug(f"{date} | ⚠️ Signal degenerate — empty or zero std")
            return pd.Series(0.0, index=full_index, name="mean_reversion_signal")

        normalized = normalize_signal(raw_score)

        strong = normalized[normalized.abs() >= self.threshold]

        self.logger.debug(
            f"{date} | Signal stats: mean={normalized.mean():.4f}, std={normalized.std():.4f}, n_strong={len(strong)}"
        )

        if self.verbose and not strong.empty:
            self.logger.info(f"{date} | Top buys: {strong.nsmallest(3).to_dict()}")
            self.logger.info(f"{date} | Top sells: {strong.nlargest(3).to_dict()}")

        signal_out = pd.Series(0.0, index=full_index, name="mean_reversion_signal")
        signal_out.loc[strong.index] = strong
        return signal_out
