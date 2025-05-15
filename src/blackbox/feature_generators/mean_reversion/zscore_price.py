import pandas as pd

from blackbox.feature_generators.base import (BaseFeatureGenerator,
                                              register_feature)


@register_feature("zscore_price")
class ZScorePriceFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20):
        self.period = period

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError(
                "Input data must have MultiIndex with levels: ['date', 'symbol']"
            )

        close = data["close"]
        grouped = close.groupby(level="symbol")

        rolling_mean = grouped.transform(lambda x: x.rolling(self.period).mean())
        rolling_std = grouped.transform(lambda x: x.rolling(self.period).std())
        rolling_std = rolling_std.replace(0, pd.NA)  # avoid divide-by-zero

        zscore = ((close - rolling_mean) / rolling_std).rename(
            f"zscore_price_{self.period}"
        )
        zscore.index = data.index

        return zscore.to_frame()
