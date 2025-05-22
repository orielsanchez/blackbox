import numpy as np
import pandas as pd

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("adx")
class ADXFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 14):
        super().__init__()
        self.period = period

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Input data must include {required_cols}.")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError("Input data must have MultiIndex with ['date', 'symbol'].")

        df = data[["high", "low", "close"]].reset_index()

        def compute_adx(df_symbol: pd.DataFrame) -> pd.DataFrame:
            df_symbol = df_symbol.sort_values("date").copy()
            high, low, close = df_symbol["high"], df_symbol["low"], df_symbol["close"]

            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(self.period).mean()

            up_move = high.diff()
            down_move = -low.diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            plus_di = (
                100
                * pd.Series(plus_dm, index=high.index).rolling(self.period).sum()
                / atr
            )
            minus_di = (
                100
                * pd.Series(minus_dm, index=high.index).rolling(self.period).sum()
                / atr
            )

            dx_raw = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
            dx = pd.Series(dx_raw, index=high.index).replace([np.inf, -np.inf], 0.0)
            adx = dx.rolling(self.period).mean().fillna(0.0)

            return pd.DataFrame(
                {
                    "date": df_symbol["date"],
                    "symbol": df_symbol["symbol"],
                    f"adx_{self.period}": adx,
                }
            )

        result = df.groupby("symbol", group_keys=False).apply(compute_adx)

        # Restore MultiIndex with proper names
        result = result.set_index(["date", "symbol"]).sort_index()
        result.index.names = ["date", "symbol"]

        return result[[f"adx_{self.period}"]]
