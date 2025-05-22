import numpy as np
import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("rsi")
class RSIFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 14, col: str = "close"):
        """
        Compute the Relative Strength Index (RSI) over a rolling window.

        Args:
            period: Number of periods to calculate RSI over (default 14).
            col: Column of OHLCV data to compute RSI from (default "close").
        """
        super().__init__()
        self.period = period
        self.col = col

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.col not in data.columns:
            raise ValueError(f"Missing required column: '{self.col}'")

        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError(
                "Input must have MultiIndex with levels ['date', 'symbol']"
            )

        close = data[self.col]
        grouped = close.groupby(level="symbol")

        delta = grouped.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.groupby(level="symbol").transform(
            lambda x: x.ewm(alpha=1 / self.period, adjust=False).mean()
        )
        avg_loss = loss.groupby(level="symbol").transform(
            lambda x: x.ewm(alpha=1 / self.period, adjust=False).mean()
        )

        rs = avg_gain / (avg_loss + 1e-8)  # avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        rsi.name = f"rsi_{self.period}"
        return rsi.to_frame()
