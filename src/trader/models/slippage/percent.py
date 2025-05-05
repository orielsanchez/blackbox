from typing import Optional

import pandas as pd

from trader.core.slippage import SlippageModel
from trader.utils.schema import \
    standardize_model_output  # assumes shared location


class PercentSlippageModel(SlippageModel):
    def __init__(self, slippage_pct: float = 0.001):  # e.g. 0.1%
        self.slippage_pct = slippage_pct

    def score(
        self, trades: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Estimate slippage cost per symbol as a percentage of price.

        Parameters:
            trades: DataFrame with ['symbol', 'shares', 'price']
            timestamp: optional for future compatibility

        Returns:
            DataFrame with ['symbol', 'slippage']
        """
        required_cols = {"symbol", "shares", "price"}
        if not required_cols.issubset(trades.columns):
            raise ValueError(f"Missing columns: {required_cols - set(trades.columns)}")

        trades = trades.copy()
        trades["shares"] = trades["shares"].fillna(0)
        trades["price"] = trades["price"].fillna(0)

        trades["slippage"] = (
            trades["shares"].abs() * trades["price"] * self.slippage_pct
        )

        return standardize_model_output(
            trades[["symbol", "slippage"]],
            required_cols=["symbol", "slippage"],
            name="PercentSlippageModel",
        )
