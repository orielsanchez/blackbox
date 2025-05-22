import numpy as np
import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("atr")
class TrueRangeVolatilityFeature(BaseFeatureGenerator):
    def __init__(self, window: int = 14):
        """
        Computes rolling average of True Range (ATR-style volatility) over a window.

        Args:
            window: Number of periods for rolling average.
        """
        super().__init__()
        self.window = window

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = ["high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Input data must include columns: {required_cols}")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError("Input data must have MultiIndex with levels: ['date', 'symbol']")

        high = data["high"]
        low = data["low"]
        grouped = data.groupby(level="symbol")
        prev_close = grouped["close"].shift(1)

        # Compute True Range
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Rolling mean of true range = ATR
        atr = true_range.groupby(level="symbol").transform(lambda x: x.rolling(self.window).mean())
        atr.name = f"true_range_volatility_{self.window}"
        atr.index = data.index

        return atr.to_frame()
