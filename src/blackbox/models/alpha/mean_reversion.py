import pandas as pd

from blackbox.models.alpha.base import FeatureAwareAlphaModel
from blackbox.utils.context import get_logger


class MeanReversionAlphaModel(FeatureAwareAlphaModel):
    name = "mean_reversion"

    def __init__(
        self,
        window: int = 20,
        threshold: float = 0.01,
        features: list[dict] = None,
    ):
        super().__init__(features)
        self.window = window
        self.threshold = threshold

    def predict(self, snapshot: dict) -> pd.Series:
        """Alias for generate to support standard ML interface"""
        return self.generate(snapshot)

    def generate(self, snapshot: dict) -> pd.Series:
        logger = get_logger()
        logger.info(f"MeanReversionAlphaModel: Processing snapshot for {snapshot['date']}")

        # Get feature matrix for this date
        feature_matrix: pd.DataFrame = self.get_feature_matrix_for(snapshot)

        if feature_matrix is None or feature_matrix.empty:
            logger.warning(f"No features available for {snapshot['date']}")
            return pd.Series(dtype=float, index=[])

        logger.info(f"Features available for {len(feature_matrix)} symbols")
        logger.debug(f"Feature columns: {feature_matrix.columns}")

        # Verify feature values are reasonable
        has_valid_features = False
        for col in feature_matrix.columns:
            col_stats = {
                "min": feature_matrix[col].min(),
                "max": feature_matrix[col].max(),
                "mean": feature_matrix[col].mean(),
                "nonzero": (feature_matrix[col] != 0).sum(),
                "null": feature_matrix[col].isnull().sum(),
            }
            logger.debug(f"Feature stats for {col}: {col_stats}")
            if col_stats["nonzero"] > 0:
                has_valid_features = True

        if not has_valid_features:
            logger.warning("No valid non-zero feature values found")
            return pd.Series(dtype=float, index=[])

        # Get prices for normalization
        prices = snapshot["prices"]
        logger.debug(f"Prices available for {len(prices)} symbols")

        common_symbols = feature_matrix.index.intersection(prices.index)

        if len(common_symbols) == 0:
            logger.warning(f"No common symbols between features and prices")
            # Debug: Show a few symbols from each to help diagnose
            logger.debug(f"Feature symbols (sample): {list(feature_matrix.index)[:10]}")
            logger.debug(f"Price symbols (sample): {list(prices.index)[:10]}")
            return pd.Series(dtype=float, index=[])

        logger.info(f"Common symbols: {len(common_symbols)}")

        # Extract Z-score features
        zscore_columns = [col for col in feature_matrix.columns if "zscore" in col]

        if not zscore_columns:
            logger.warning(f"No Z-score features found in {feature_matrix.columns}")
            return pd.Series(dtype=float, index=[])

        logger.info(f"Found Z-score features: {zscore_columns}")

        # Get subset with common symbols and z-score columns
        feature_subset = feature_matrix.loc[common_symbols, zscore_columns]

        # Debug: How many non-null values?
        non_null_counts = feature_subset.notna().sum()
        logger.debug(f"Non-null counts in feature subset: {non_null_counts.to_dict()}")

        # Drop symbols with all NaN features
        valid_symbols = feature_subset.dropna(how="all").index
        logger.debug(f"Symbols with valid features: {len(valid_symbols)}/{len(common_symbols)}")

        if len(valid_symbols) == 0:
            logger.warning("All symbols have NaN features")
            return pd.Series(dtype=float, index=[])

        # Keep only symbols with valid features
        feature_subset = feature_subset.loc[valid_symbols]

        # Generate signals: negative Z-score = buy, positive = sell
        signals = -feature_subset.mean(axis=1)

        # Log raw signals before thresholding
        if not signals.empty:
            signal_stats = {
                "min": signals.min(),
                "max": signals.max(),
                "mean": signals.mean(),
                "count": len(signals),
            }
            logger.debug(f"Signal stats before thresholding: {signal_stats}")
            logger.info(f"Raw signals: {signals.to_dict()}")

        # Apply threshold from config
        actual_threshold = self.threshold
        logger.info(f"Using threshold: {actual_threshold} (config: {self.threshold})")

        # Threshold signals to avoid noise
        signals = signals[abs(signals) > actual_threshold]

        logger.info(f"Generated {len(signals)} signals after thresholding")
        if len(signals) > 0:
            top_signals = signals.sort_values().head(3)
            bottom_signals = signals.sort_values(ascending=False).head(3)
            logger.info(f"Top buy signals: {top_signals.to_dict()}")
            logger.info(f"Top sell signals: {bottom_signals.to_dict()}")

        return signals
