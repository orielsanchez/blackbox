import pandas as pd

from trader.core.alpha import AlphaModel
from trader.utils.schema import standardize_model_output


class MomentumAlphaModel(AlphaModel):
    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def score(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        """
        Parameters:
            data: DataFrame with columns ['symbol', 'timestamp', 'price']
            timestamp: the time to compute scores for

        Returns:
            DataFrame with index 'symbol' and column 'alpha_score'
        """
        if not {"symbol", "timestamp", "price"}.issubset(data.columns):
            raise ValueError("Data must contain 'symbol', 'timestamp', 'price' columns")

        data = data[data["timestamp"] <= timestamp]
        data = data.sort_values(["symbol", "timestamp"])

        scores = {}
        for symbol, group in data.groupby("symbol"):
            if len(group) < self.lookback + 1:
                scores[symbol] = 0.0  # not enough history
                continue

            group = group.tail(self.lookback + 1)
            old_price = group.iloc[0]["price"]
            new_price = group.iloc[-1]["price"]

            if old_price and pd.notnull(old_price) and pd.notnull(new_price):
                scores[symbol] = (new_price / old_price) - 1.0
            else:
                scores[symbol] = 0.0

        df = pd.DataFrame.from_dict(
            scores, orient="index", columns=["alpha_score"]
        ).reset_index(names="symbol")
        return standardize_model_output(
            df, required_cols=["symbol", "alpha_score"], name="MomentumAlphaModel"
        )
