import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("momentum")
class MomentumFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20):
        """
        Computes n-period percentage change in closing prices (momentum).
        """
        self.period = period

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "close" not in data.columns:
            raise ValueError("Input data must include a 'close' column.")
        if not isinstance(data.index, pd.MultiIndex) or set(data.index.names) != {
            "date",
            "symbol",
        }:
            raise ValueError(
                "Input data must have a MultiIndex with levels ['date', 'symbol']."
            )

        close = data["close"]
        returns = close.groupby(level="symbol").pct_change(periods=self.period)
        returns.name = f"momentum_{self.period}"
        return returns.to_frame().fillna(0.0).sort_index()
