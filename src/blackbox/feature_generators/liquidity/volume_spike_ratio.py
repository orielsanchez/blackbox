import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("volume_spike_ratio")
class VolumeSpikeRatioFeature(BaseFeatureGenerator):
    def __init__(self, window: int = 20):
        """
        Computes the ratio of current volume to rolling mean volume.

        Args:
            window: Lookback window for computing average volume.
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

        rolling_avg = grouped.transform(lambda x: x.rolling(self.window).mean())
        spike_ratio = (volume / rolling_avg).replace([float("inf"), -float("inf")], pd.NA)

        spike_ratio.name = f"volume_spike_ratio_{self.window}"
        spike_ratio.index = data.index

        return spike_ratio.to_frame()
