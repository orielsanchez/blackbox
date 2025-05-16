import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("true_range")
class TrueRangeFeature(BaseFeatureGenerator):
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.groupby(level=1).shift(1)

        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        return tr.rename("true_range").to_frame()
