import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("ewma_volatility")
class EWMAVolatilityFeature(BaseFeatureGenerator):
    def __init__(self, span: int = 20, col: str = "close", output: str = None):
        """
        Computes EWMA volatility (std dev of exponentially weighted returns).

        Args:
            span: The decay span for the EWM.
            col: Column to use for returns (default: close).
            output: Optional custom name for output column.
        """
        super().__init__()
        self.span = span
        self.col = col
        self.output = output or f"ewm_vol_{span}"

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError(
                "Input data must have MultiIndex with levels: ['date', 'symbol']"
            )

        if self.col not in data.columns:
            raise ValueError(f"Missing column '{self.col}' in input data")

        prices = data[self.col]
        grouped = prices.groupby(level="symbol")

        returns = grouped.pct_change()
        vol = returns.groupby(level="symbol").transform(
            lambda x: x.ewm(span=self.span, adjust=False).std()
        )

        vol.name = self.output
        return vol.to_frame()
