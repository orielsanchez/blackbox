import pandas as pd

from trader.core.alpha import AlphaModel


class MomentumAlphaModel(AlphaModel):
    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        latest = data.groupby("symbol").last().copy()
        latest["alpha_score"] = (
            data.groupby("symbol")["close"]
            .apply(lambda x: x.pct_change(10))
            .groupby("symbol")
            .last()
        )
        return latest[["alpha_score"]].dropna()
