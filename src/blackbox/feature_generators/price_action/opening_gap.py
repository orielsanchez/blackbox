import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("opening_gap")
class OpeningGapFeature(BaseFeatureGenerator):
    def __init__(self):
        """
        Computes the opening gap as a percentage difference
        between today's open and previous day's close:
        (open_t / close_{t-1}) - 1
        """
        super().__init__()

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        required = ["open", "close"]
        if not all(col in data.columns for col in required):
            raise ValueError(f"Input data must include {required}.")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError("Input data must have MultiIndex with levels: ['date', 'symbol'].")

        open_ = data["open"]
        grouped = data.groupby(level="symbol")
        prev_close = grouped["close"].shift(1)

        gap = (open_ / prev_close) - 1.0
        gap.name = "opening_gap"
        gap.index = data.index

        return gap.to_frame()
