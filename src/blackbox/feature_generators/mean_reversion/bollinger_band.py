import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("bollinger_band")
class BollingerBandFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        mean = close.groupby(level=1).rolling(self.period).mean().droplevel(0)
        std = close.groupby(level=1).rolling(self.period).std().droplevel(0)
        upper = mean + self.std_dev * std
        lower = mean - self.std_dev * std
        norm_band = ((close - lower) / (upper - lower)).rename(
            f"bollinger_norm_{self.period}"
        )
        return norm_band.to_frame()
