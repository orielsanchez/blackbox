import pandas as pd

from trader.core.alpha import AlphaModel
from trader.utils.schema import standardize_model_output


class MomentumAlphaModel(AlphaModel):
    def __init__(self, lookback: int = 20, min_lookback: int = 3, strict: bool = False):
        """
        lookback: default lookback window (must be >= 1)
        min_lookback: minimum lookback window to fall back to (>=1)
        strict: if True, assigns 0 if full lookback not met; else adaptively shrinks
        """
        self.lookback = lookback
        self.min_lookback = min_lookback
        self.strict = strict

    def score(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        if not {"symbol", "timestamp", "price"}.issubset(data.columns):
            raise ValueError("Data must contain 'symbol', 'timestamp', 'price' columns")

        data = data[data["timestamp"] <= timestamp]
        data = data.sort_values(["symbol", "timestamp"])

        scores = {}
        for symbol, group in data.groupby("symbol"):
            n = len(group)

            if self.strict:
                if n < self.lookback + 1:
                    scores[symbol] = 0.0
                    continue
                used_lookback = self.lookback
            else:
                used_lookback = max(min(self.lookback, n - 1), self.min_lookback)
                if n < used_lookback + 1:
                    scores[symbol] = 0.0
                    continue

            group = group.tail(used_lookback + 1)
            old_price = group.iloc[0]["price"]
            new_price = group.iloc[-1]["price"]

            if pd.notnull(old_price) and pd.notnull(new_price) and old_price != 0:
                momentum = (new_price / old_price) - 1.0
                scores[symbol] = momentum
            else:
                print(f"⚠️ {symbol}: invalid prices (old={old_price}, new={new_price})")
                scores[symbol] = 0.0

        df = pd.DataFrame.from_dict(
            scores, orient="index", columns=["alpha_score"]
        ).reset_index(names="symbol")

        if df["alpha_score"].abs().sum() == 0.0:
            print("⚠️ All alpha scores are 0. Check data quality or time window.")

        return standardize_model_output(
            df, required_cols=["symbol", "alpha_score"], name="MomentumAlphaModel"
        )
