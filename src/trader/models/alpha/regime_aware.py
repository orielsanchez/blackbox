import numpy as np
import pandas as pd

from trader.core.alpha import AlphaModel
from trader.utils.schema import standardize_model_output


class RegimeAwareAlphaModel(AlphaModel):
    def __init__(
        self,
        momentum_model,
        mean_rev_model,
        window: int = 30,
        trend_threshold: float = 0.7,
    ):
        self.momentum_model = momentum_model
        self.mean_rev_model = mean_rev_model
        self.window = window
        self.trend_threshold = trend_threshold  # based on R²

    def is_trending(self, market_df: pd.DataFrame) -> bool:
        df = market_df.sort_values("timestamp").tail(self.window).copy()
        df["log_price"] = np.log(df["price"])
        df["t"] = range(len(df))
        coeffs = np.polyfit(df["t"], df["log_price"], 1)
        fitted = np.polyval(coeffs, df["t"])
        ss_res = ((df["log_price"] - fitted) ** 2).sum()
        ss_tot = ((df["log_price"] - df["log_price"].mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot if ss_tot > 0 else 0)
        return r_squared >= self.trend_threshold

    def score(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        market_mean = data.groupby("timestamp")["price"].mean().reset_index()
        trending = self.is_trending(market_mean[market_mean["timestamp"] <= timestamp])

        model = self.momentum_model if trending else self.mean_rev_model
        result = model.score(data, timestamp)
        result["regime"] = "trend" if trending else "mean_reversion"
        return standardize_model_output(
            result,
            required_cols=["symbol", "alpha_score"],
            name="RegimeAwareAlphaModel",
        )
