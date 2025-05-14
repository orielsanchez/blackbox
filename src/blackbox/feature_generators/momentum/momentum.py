import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("momentum")
class MomentumFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20):
        self.period = period

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        returns = close.groupby(level=1).pct_change(periods=self.period)
        return returns.rename(f"momentum_{self.period}d").to_frame()
