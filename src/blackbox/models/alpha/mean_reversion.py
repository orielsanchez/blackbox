import logging

import pandas as pd

from blackbox.models.interfaces import AlphaModel
from blackbox.utils.context import get_feature_matrix, get_logger


class MeanReversionAlphaModel(AlphaModel):
    name = "mean_reversion"

    def __init__(self, window: int = 20, threshold: float = 0.01, features: list[dict] = None):
        self.window = window
        self.threshold = threshold
        self.logger = get_logger()

    def generate(self, snapshot: dict) -> pd.Series:
        date: pd.Timestamp = snapshot["date"]
        feature_matrix = get_feature_matrix()

        # Feature column names expected
        zscore_col = f"zscore_price_{self.window}"
        bollinger_col = f"bollinger_norm_{self.window}"
        required_cols = [zscore_col, bollinger_col]

        # Check required columns exist
        for col in required_cols:
            if col not in feature_matrix.columns:
                msg = f"❌ Required feature column '{col}' missing from feature matrix"
                self.logger.error(msg)
                raise ValueError(msg)

        # Extract features for current date
        try:
            today_df = feature_matrix.loc[date]
        except KeyError:
            self.logger.warning(f"⚠️ No feature data for {date.date()}")
            return pd.Series(dtype=float)

        # Flatten symbol index if MultiIndex
        if isinstance(today_df.index, pd.MultiIndex):
            today_df.index = today_df.index.get_level_values("symbol")

        # Drop rows with NaNs in either feature column
        today_df = today_df.dropna(subset=required_cols)

        if today_df.empty:
            self.logger.warning(
                f"{date.date()} | All rows dropped due to NaNs in required features"
            )
            return pd.Series(dtype=float)

        # Mean reversion signal: the lower the zscore/band, the more mean-reverting
        scores = -0.6 * today_df[zscore_col] - 0.4 * today_df[bollinger_col]

        # Apply threshold
        scores = scores.where(scores.abs() > self.threshold, 0.0).fillna(0.0)

        # Check for duplicated symbols
        if scores.index.duplicated().any():
            dupes = scores.index[scores.index.duplicated()].tolist()
            self.logger.warning(f"{date.date()} | Duplicate symbols in score: {dupes}")
            scores = scores[~scores.index.duplicated(keep="first")]

        self.logger.debug(f"{date.date()} | Top signals: {scores[scores != 0].head().to_dict()}")

        return scores.sort_index()
