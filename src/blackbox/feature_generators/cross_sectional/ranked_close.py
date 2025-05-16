import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("ranked_close")
class RankedCloseFeature(BaseFeatureGenerator):
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        def rank_close(group: pd.DataFrame) -> pd.Series:
            return group["close"].rank(pct=True).rename("ranked_close")

        ranked = data.groupby(level=0).apply(rank_close)
        return ranked.to_frame()
