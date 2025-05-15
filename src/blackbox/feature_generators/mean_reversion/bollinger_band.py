import pandas as pd

from blackbox.feature_generators.base import (BaseFeatureGenerator,
                                              register_feature)


@register_feature("bollinger_band")
class BollingerBandFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError(
                "Input data must have MultiIndex with levels: ['date', 'symbol']"
            )

        close = data["close"]

        # Compute rolling mean and std per symbol
        grouped = close.groupby(level="symbol")
        mean = grouped.transform(lambda x: x.rolling(self.period).mean())
        std = grouped.transform(lambda x: x.rolling(self.period).std())

        upper = mean + self.std_dev * std
        lower = mean - self.std_dev * std

        # Avoid divide-by-zero errors
        spread = upper - lower
        spread = spread.replace(0, pd.NA)

        # Compute normalized band position
        norm_band = ((close - lower) / spread).rename(f"bollinger_norm_{self.period}")
        norm_band.index = data.index

        # Output as DataFrame with consistent MultiIndex
        return norm_band.to_frame()
