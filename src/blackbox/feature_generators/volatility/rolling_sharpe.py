import numpy as np
import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("rolling_sharpe")
class RollingSharpeFeature(BaseFeatureGenerator):
    def __init__(self, window: int = 20):
        """
        Computes rolling Sharpe ratio using log returns.

        Args:
            window: Number of periods for the rolling window.
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

        # Compute log returns
        log_returns = grouped.transform(lambda x: np.log(x / x.shift(1)))

        # Rolling mean and std
        rolling_mean = log_returns.groupby(level="symbol").transform(
            lambda x: x.rolling(self.window).mean()
        )
        rolling_std = log_returns.groupby(level="symbol").transform(
            lambda x: x.rolling(self.window).std()
        )

        sharpe = (rolling_mean / rolling_std).replace([np.inf, -np.inf], pd.NA).fillna(0.0)
        sharpe.name = f"rolling_sharpe_{self.window}"
        sharpe.index = data.index

        return sharpe.to_frame()
