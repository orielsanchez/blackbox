# src/blackbox/models/alpha/mean_reversion.py

import pandas as pd

from blackbox.models.alpha.base import FeatureAwareAlphaModel


class MeanReversionAlphaModel(FeatureAwareAlphaModel):
    name = "mean_reversion"

    def __init__(
        self, window: int = 20, threshold: float = 0.01, features: list[dict] = None
    ):
        super().__init__(features)
        self.window = window
        self.threshold = threshold

    def generate(self, snapshot: dict) -> pd.Series:
        date: pd.Timestamp = snapshot["date"]
        feature_matrix = self.get_feature_matrix_for(snapshot)

        zscore_col = self.feature_columns.get("zscore")
        bollinger_col = self.feature_columns.get("bollinger")
        required_cols = [zscore_col, bollinger_col]

        for col in required_cols:
            if col not in feature_matrix.columns:
                msg = f"❌ Required feature column '{col}' missing from feature matrix"
                self.logger.error(msg)
                raise ValueError(msg)

        try:
            today_df = feature_matrix.loc[date]
        except KeyError:
            self.logger.warning(f"⚠️ No feature data for {date.date()}")
            return pd.Series(dtype=float)

        if isinstance(today_df.index, pd.MultiIndex):
            today_df.index = today_df.index.get_level_values("symbol")

        today_df = today_df.dropna(subset=required_cols)

        if today_df.empty:
            self.logger.warning(
                f"{date.date()} | All rows dropped due to NaNs in required features"
            )
            return pd.Series(dtype=float)

        scores = -0.6 * today_df[zscore_col] - 0.4 * today_df[bollinger_col]
        scores = scores.where(scores.abs() > self.threshold, 0.0).fillna(0.0)

        if scores.index.duplicated().any():
            dupes = scores.index[scores.index.duplicated()].tolist()
            self.logger.warning(f"{date.date()} | Duplicate symbols in score: {dupes}")
            scores = scores[~scores.index.duplicated(keep="first")]

        self.logger.debug(
            f"{date.date()} | Top signals: {scores[scores != 0].head().to_dict()}"
        )

        return scores.sort_index()
