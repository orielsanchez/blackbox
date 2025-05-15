import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("ema_crossover")
class EMACrossoverFeature(BaseFeatureGenerator):
    def __init__(self, short: int = 10, long: int = 50):
        self.short = short
        self.long = long

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        short_ema = close.groupby(level=1).transform(lambda x: x.ewm(span=self.short).mean())
        long_ema = close.groupby(level=1).transform(lambda x: x.ewm(span=self.long).mean())
        diff = (short_ema - long_ema).rename(f"ema_{self.short}_{self.long}_diff")
        return diff.to_frame()
