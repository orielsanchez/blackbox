import pandas as pd

from blackbox.models.interfaces import TransactionCostModel


class FixedTransactionCostModel(TransactionCostModel):
    name = "fixed"

    def __init__(self, slippage: float = 0.0001, commission: float = 0.0001):
        self.slippage = slippage  # proportional to trade value
        self.commission = commission  # also proportional

    def estimate(self, trades: pd.Series, prices: pd.Series) -> pd.Series:
        """
        Returns estimated cost per symbol as a fraction of portfolio value.
        Cost = |trade| * price * (slippage + commission)
        """
        cost_rate = self.slippage + self.commission
        return (trades.abs() * prices * cost_rate).fillna(0.0)

    def adjust(self, signals: pd.Series, current: pd.Series) -> pd.Series:
        """
        Adjust signals by subtracting cost from raw signal strength.
        """
        trades = signals.subtract(current, fill_value=0.0)
        # Assume price = 1 for normalizing cost impact; or inject actual prices here
        dummy_prices = pd.Series(1.0, index=trades.index)
        cost_penalty = self.estimate(trades, dummy_prices)
        return signals - cost_penalty
