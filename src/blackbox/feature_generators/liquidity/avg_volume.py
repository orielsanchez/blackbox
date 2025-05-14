import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("avg_volume")
class AvgVolumeFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20):
        self.period = period

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        vol = data["volume"].groupby(level=1).rolling(self.period).mean().droplevel(0)
        return vol.rename(f"avg_volume_{self.period}d").to_frame()
