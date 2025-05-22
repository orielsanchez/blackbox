import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("rolling_mean")
class RollingMeanFeature(BaseFeatureGenerator):
    def __init__(self, window: int = 20):
        """
        Computes the rolling mean of close prices over a specified window.

        Args:
            window: Number of periods for the rolling average.
        """
        super().__init__()
        self.window = window

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "close" not in data.columns:
            raise ValueError("Input data must include 'close' column.")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError("Input data must have MultiIndex with levels: ['date', 'symbol']")

        close = data["close"]
        grouped = close.groupby(level="symbol")

        rolling_mean = grouped.transform(lambda x: x.rolling(self.window).mean())
        rolling_mean.name = f"rolling_mean_{self.window}"
        rolling_mean.index = data.index

        return rolling_mean.to_frame()
