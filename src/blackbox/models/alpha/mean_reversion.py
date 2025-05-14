import pandas as pd

from blackbox.models.interfaces import AlphaModel
from blackbox.utils.context import get_feature_matrix, get_logger


class MeanReversionAlphaModel(AlphaModel):
    name = "mean_reversion"

    def __init__(
        self, window: int = 20, threshold: float = 0.01, features: list[dict] = None
    ):
        self.window = window
        self.threshold = threshold
        self.logger = get_logger()

    def generate(self, snapshot: dict) -> pd.Series:
        date: pd.Timestamp = snapshot["date"]
        feature_matrix = get_feature_matrix()

        z_col = f"zscore_{self.window}d"
        b_col = f"bollinger_norm_{self.window}"

        assert z_col in feature_matrix.columns, f"{z_col} missing"
        assert b_col in feature_matrix.columns, f"{b_col} missing"

        if date not in feature_matrix.index.get_level_values(0):
            self.logger.warning(f"⚠️ No features for {date.date()}")
            return pd.Series(dtype=float)

        try:
            today_features = feature_matrix.loc[date]
        except KeyError:
            self.logger.warning(f"⚠️ No features for {date.date()} (KeyError)")
            return pd.Series(dtype=float)

        # ✅ Flatten index to symbols if MultiIndex
        if isinstance(today_features.index, pd.MultiIndex):
            today_features.index = today_features.index.get_level_values("symbol")

        # Validate columns exist
        if z_col not in today_features.columns or b_col not in today_features.columns:
            self.logger.warning(f"⚠️ Missing expected features: {z_col} or {b_col}")
            return pd.Series(dtype=float)

        self.logger.debug(
            f"{date.date()} feature snapshot: {today_features[[z_col, b_col]].describe().to_dict()}"
        )

        score = -0.6 * today_features[z_col] - 0.4 * today_features[b_col]
        score = score.where(score.abs() > self.threshold, 0)

        # ✅ Remove any lingering NaNs
        score = score.fillna(0)

        # ✅ Final check for duplicates
        if score.index.duplicated().any():
            dupes = score.index[score.index.duplicated()].tolist()
            self.logger.warning(f"⚠️ Alpha output contains duplicate symbols: {dupes}")
            score = score[~score.index.duplicated(keep="first")]

        self.logger.debug(
            f"{date.date()} score sample: {score[score != 0].head().to_dict()}"
        )

        return score.sort_index()
