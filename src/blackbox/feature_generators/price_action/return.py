import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("return")
class ReturnFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 1):
        """
        Computes N-day arithmetic returns: (close_t / close_{t-N}) - 1.

        Args:
            period: Number of trading days between prices.
        """
        super().__init__()
        self.period = period

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "close" not in data.columns:
            raise ValueError("Input data must include 'close' column.")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError("Input data must have MultiIndex with levels: ['date', 'symbol']")

        close = data["close"]
        grouped = close.groupby(level="symbol")

        returns = grouped.transform(lambda x: x / x.shift(self.period) - 1.0)
        returns.name = f"return_{self.period}d"
        returns.index = data.index

        return returns.to_frame()
