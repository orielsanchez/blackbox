import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("drawdown")
class DrawdownFeature(BaseFeatureGenerator):
    def __init__(self, window: int = 252):
        """
        Computes rolling drawdown: (close - rolling_max) / rolling_max.

        Args:
            window: Lookback window to compute the rolling peak.
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

        rolling_max = grouped.transform(lambda x: x.rolling(self.window).max())

        drawdown = (close - rolling_max) / rolling_max
        drawdown = drawdown.fillna(0.0)  # safe default: 0 when rolling_max is NA
        drawdown.name = f"drawdown_{self.window}"
        drawdown.index = data.index

        return drawdown.to_frame()
