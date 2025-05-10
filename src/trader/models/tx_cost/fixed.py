from typing import Optional

import pandas as pd

from trader.core.tx_cost import TransactionCostModel
from trader.utils.schema import standardize_model_output


class FixedCostModel(TransactionCostModel):
    def __init__(self, cost_per_share: float = 0.01):
        self.cost_per_share = cost_per_share

    def score(
        self, trades: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Estimate transaction costs based on trade size.

        Parameters:
            trades: DataFrame with ['symbol', 'shares']
            timestamp: optional, unused in fixed cost model

        Returns:
            DataFrame with ['symbol', 'tx_cost']
        """
        required_cols = {"symbol", "shares"}
        if not required_cols.issubset(trades.columns):
            raise ValueError(f"Missing columns: {required_cols - set(trades.columns)}")

        trades = trades.copy()
        trades["shares"] = trades["shares"].fillna(0)
        trades["tx_cost"] = trades["shares"].abs() * self.cost_per_share

        return standardize_model_output(
            trades[["symbol", "tx_cost"]],
            required_cols=["symbol", "tx_cost"],
            name="FixedCostModel",
        )
