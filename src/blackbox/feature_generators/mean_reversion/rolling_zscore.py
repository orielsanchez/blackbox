import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("rolling_zscore")
class RollingZScoreFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20):
        """
        Computes the z-score of close price relative to its rolling mean and std.

        Args:
            period: Number of periods for the rolling window.
        """
        super().__init__()
        self.period = int(period)

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "close" not in data.columns:
            raise ValueError("Input data must include 'close' column.")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError(
                "Input data must have MultiIndex with levels: ['date', 'symbol']"
            )

        close = data["close"]
        grouped = close.groupby(level="symbol")

        rolling_mean = grouped.transform(lambda x: x.rolling(self.period).mean())
        rolling_std = grouped.transform(lambda x: x.rolling(self.period).std())

        zscore = ((close - rolling_mean) / rolling_std).fillna(0.0)
        zscore.name = f"rolling_zscore_{self.period}"
        zscore.index = data.index

        return zscore.to_frame()
