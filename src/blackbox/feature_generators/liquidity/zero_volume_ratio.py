import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("zero_volume_ratio")
class ZeroVolumeRatioFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20):
        self.period = period

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        zero_days = (
            (data["volume"] == 0)
            .groupby(level=1)
            .rolling(self.period)
            .sum()
            .droplevel(0)
        )
        ratio = (zero_days / self.period).rename(f"zero_volume_ratio_{self.period}")
        return ratio.to_frame()
