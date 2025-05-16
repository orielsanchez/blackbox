import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("zscore_price")
class ZScorePriceFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20):
        super().__init__()
        self.period = period

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError(
                "Input data must have MultiIndex with levels: ['date', 'symbol']"
            )

        close = data["close"]
        grouped = close.groupby(level="symbol")

        mean = grouped.transform(lambda x: x.rolling(self.period).mean())
        std = grouped.transform(lambda x: x.rolling(self.period).std()).replace(
            0, pd.NA
        )

        zscore = ((close - mean) / std).rename(f"zscore_price_{self.period}")
        zscore.index = data.index

        return zscore.to_frame()
