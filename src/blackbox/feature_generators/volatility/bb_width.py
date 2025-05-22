import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("bb_width")
class BollingerBandWidthFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Computes Bollinger Band Width: (upper - lower) / mid

        Args:
            period: Rolling window for SMA and std deviation.
            std_dev: Multiplier for standard deviation to define bands.
        """
        super().__init__()
        self.period = period
        self.std_dev = std_dev

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "close" not in data.columns:
            raise ValueError("Input data must include 'close' column.")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError("Input data must have MultiIndex with levels: ['date', 'symbol']")

        close = data["close"]
        grouped = close.groupby(level="symbol")

        mean = grouped.transform(lambda x: x.rolling(self.period).mean())
        std = grouped.transform(lambda x: x.rolling(self.period).std())

        upper = mean + self.std_dev * std
        lower = mean - self.std_dev * std
        width = (upper - lower) / mean.replace(0, pd.NA)

        width.name = f"bb_width_{self.period}"
        width.index = data.index

        return width.to_frame()
