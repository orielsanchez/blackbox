import pandas as pd

from trader.core.alpha import AlphaModel
from trader.utils.schema import standardize_model_output


class MeanReversionAlphaModel(AlphaModel):
    def __init__(self, window: int = 10):
        """
        window: rolling window length for mean reversion
        """
        self.window = window

    def score(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        if not {"symbol", "timestamp", "price"}.issubset(data.columns):
            raise ValueError("Data must contain 'symbol', 'timestamp', 'price' columns")

        data = data[data["timestamp"] <= timestamp].copy()
        data = data.sort_values(["symbol", "timestamp"])

        scores = {}
        for symbol, group in data.groupby("symbol"):
            if len(group) < self.window:
                scores[symbol] = 0.0
                continue

            recent = group.tail(self.window)
            mean_price = recent["price"].mean()
            current_price = recent.iloc[-1]["price"]

            if pd.notnull(current_price) and pd.notnull(mean_price) and mean_price != 0:
                # Positive if price < mean (i.e. expected reversion upward)
                scores[symbol] = (mean_price - current_price) / mean_price
            else:
                scores[symbol] = 0.0

        df = pd.DataFrame.from_dict(
            scores, orient="index", columns=["alpha_score"]
        ).reset_index(names="symbol")

        return standardize_model_output(
            df, required_cols=["symbol", "alpha_score"], name="MeanReversionAlphaModel"
        )
