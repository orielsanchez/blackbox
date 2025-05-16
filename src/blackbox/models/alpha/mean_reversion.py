from typing import Dict, List, Optional

import pandas as pd

from blackbox.models.alpha.base import BaseAlphaModel
from blackbox.utils.context import get_logger


class MeanReversionAlphaModel(BaseAlphaModel):
    """Mean reversion alpha model.

    Generates signals based on price deviations from the mean.
    """

    name = "mean_reversion"

    def __init__(
        self,
        threshold: float = 0.5,
        features: Optional[List[Dict]] = None,
        universe: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        """Initialize the mean reversion model.

        Args:
            threshold: Signal threshold
            features: Feature specifications
            universe: Trading universe
            verbose: Whether to output detailed logs
        """
        super().__init__(verbose=verbose)
        self.threshold = threshold
        self.features = features or []
        self.universe = universe or []
        self.logger = get_logger()

    def generate(self, snapshot: Dict) -> pd.Series:
        """Generate mean reversion signals.

        Args:
            snapshot: Market data snapshot

        Returns:
            pd.Series: Alpha signals
        """
        date = pd.to_datetime(snapshot["date"]).normalize()
        self.log_info(f"MeanReversionAlphaModel: Processing snapshot for {date}")

        if "feature_vector" not in snapshot:
            self.logger.warning(f"No features available for {date}")
            return pd.Series(dtype=float)

        feature_vector = snapshot["feature_vector"]

        # Only log this in verbose mode to reduce output
        self.log_debug(f"Features available for {len(feature_vector)} symbols")

        # Filter to our universe if specified
        if self.universe:
            feature_vector = feature_vector[feature_vector.index.isin(self.universe)]

        # Find common symbols between prices and features
        prices = snapshot["prices"]
        common_symbols = feature_vector.index.intersection(prices.index)

        # Only log this in verbose mode
        self.log_debug(f"Common symbols: {len(common_symbols)}")

        if common_symbols.empty:
            self.logger.warning("No common symbols between prices and features")
            return pd.Series(dtype=float)

        # Find Z-score features for mean reversion
        zscore_features = [f for f in self.features if "zscore" in f["name"]]
        if not zscore_features:
            self.logger.warning("No Z-score features specified")
            return pd.Series(dtype=float)

        # Get feature names with periods
        feature_names = []
        for f in zscore_features:
            period = f["params"].get("period", 20)
            feature_names.append(f"{f['name']}_{period}")

        # Only log in verbose mode
        self.log_debug(f"Found Z-score features: {feature_names}")

        # Find features in the data
        available_features = feature_vector.columns.intersection(feature_names)
        if not available_features.any():
            self.logger.warning(
                f"No matching features found. Available: {feature_vector.columns.tolist()}"
            )
            return pd.Series(dtype=float)

        # Use the first available Z-score feature
        feature_name = available_features[0]

        # Generate signals
        signals = -feature_vector[feature_name]  # Negative Z-score for mean reversion
        signals = signals.dropna()

        if self.verbose:  # Only log raw signals in verbose mode
            self.log_info(f"Raw signals: {signals.to_dict()}")

        # Apply threshold filter - these are important to always log
        threshold = self.params.get("threshold", self.threshold)
        self.log_debug(f"Using threshold: {threshold} (config: {self.threshold})")

        filtered_signals = signals[signals.abs() >= threshold]
        self.log_debug(f"Generated {len(filtered_signals)} signals after thresholding")

        if not filtered_signals.empty:
            # Always log top signals regardless of verbose mode - this is important info
            top_buys = filtered_signals.sort_values(ascending=True).head(3)
            top_sells = filtered_signals.sort_values(ascending=False).head(3)
            self.log_debug(f"Top buy signals: {top_buys.to_dict()}")
            self.log_debug(f"Top sell signals: {top_sells.to_dict()}")

        return filtered_signals
