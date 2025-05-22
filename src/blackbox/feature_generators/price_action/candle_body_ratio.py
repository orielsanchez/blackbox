import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("candle_body_ratio")
class CandleBodyRatioFeature(BaseFeatureGenerator):
    def __init__(self):
        """
        Computes the ratio of the candle body (|close - open|) to the full range (high - low).

        This helps identify strong conviction candles (large body) vs. indecision candles (small body).
        """
        super().__init__()

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        required = ["open", "close", "high", "low"]
        if not all(col in data.columns for col in required):
            raise ValueError(f"Input data must include columns: {required}")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError("Input data must have MultiIndex with levels: ['date', 'symbol'].")

        open_ = data["open"]
        close = data["close"]
        high = data["high"]
        low = data["low"]

        body = (close - open_).abs()
        range_ = (high - low).replace(0, pd.NA)  # avoid division by zero

        ratio = (body / range_).fillna(0.0)
        ratio.name = "candle_body_ratio"
        ratio.index = data.index

        return ratio.to_frame()
