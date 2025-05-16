import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("zscore_vs_universe")
class ZScoreVsUniverseFeature(BaseFeatureGenerator):
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]

        def zscore(group):
            return ((group - group.mean()) / group.std()).rename("zscore_vs_universe")

        return close.groupby(level=0).apply(zscore).to_frame()
