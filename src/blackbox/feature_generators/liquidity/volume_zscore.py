import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("volume_zscore")
class VolumeZScoreFeature(BaseFeatureGenerator):
    def __init__(self, window: int = 20):
        """
        Computes the z-score of volume relative to its rolling mean and std.

        Args:
            window: Number of periods for the rolling window.
        """
        super().__init__()
        self.window = window

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "volume" not in data.columns:
            raise ValueError("Input data must include 'volume' column.")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError("Input data must have MultiIndex with levels: ['date', 'symbol']")

        volume = data["volume"]
        grouped = volume.groupby(level="symbol")

        rolling_mean = grouped.transform(lambda x: x.rolling(self.window).mean())
        rolling_std = grouped.transform(lambda x: x.rolling(self.window).std())

        zscore = ((volume - rolling_mean) / rolling_std).fillna(0.0)
        zscore.name = f"volume_zscore_{self.window}"
        zscore.index = data.index

        return zscore.to_frame()
