import numpy as np
import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("trend_strength")
class TrendStrengthFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 20):
        super().__init__()
        self.period = period

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "close" not in data.columns:
            raise ValueError("Input data must include 'close' column.")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError("Input data must have MultiIndex with ['date', 'symbol'].")

        df = data[["close"]].reset_index()

        def compute_slope(df_symbol: pd.DataFrame) -> pd.DataFrame:
            df_symbol = df_symbol.sort_values("date").copy()
            log_price = np.log(df_symbol["close"])
            slope = log_price.rolling(self.period).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0], raw=False
            )
            return pd.DataFrame(
                {
                    "date": df_symbol["date"],
                    "symbol": df_symbol["symbol"],
                    f"trend_strength_{self.period}": slope,
                }
            )

        result = df.groupby("symbol", group_keys=False).apply(compute_slope)

        # Restore MultiIndex with proper names
        result = result.set_index(["date", "symbol"]).sort_index()
        result.index.names = ["date", "symbol"]

        return result[[f"trend_strength_{self.period}"]]
