import numpy as np
import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("log_return")
class LogReturnFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 1):
        """
        Computes N-day log returns: ln(close_t / close_{t-N}).

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

        log_returns = grouped.transform(lambda x: np.log(x / x.shift(self.period)))
        log_returns.name = f"log_return_{self.period}d"
        log_returns.index = data.index

        return log_returns.to_frame()
