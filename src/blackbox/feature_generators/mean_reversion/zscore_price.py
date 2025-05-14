import pandas as pd

from blackbox.feature_generators import logger
from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("zscore_price")
class ZScorePriceFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20):
        self.period = period
        if logger:
            logger.debug(f"📈 ZScoreFeature initialized with period={self.period}")

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        mean = close.groupby(level=1).rolling(self.period).mean().droplevel(0)
        std = close.groupby(level=1).rolling(self.period).std().droplevel(0)
        zscore = ((close - mean) / std).rename(f"zscore_{self.period}d")
        return zscore.to_frame()
