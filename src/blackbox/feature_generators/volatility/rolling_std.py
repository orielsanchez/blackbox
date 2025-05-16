import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("rolling_std")
class RollingStdFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 10):
        self.period = period

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        ret = data["close"].groupby(level=1).pct_change()
        std = ret.groupby(level=1).rolling(self.period).std().droplevel(0)
        return std.rename(f"rolling_std_{self.period}d").to_frame()
