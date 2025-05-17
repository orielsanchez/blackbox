import pandas as pd

from blackbox.models.interfaces import TransactionCostModel


class FixedTransactionCostModel(TransactionCostModel):
    """
    Transaction cost model that penalizes signals proportionally based on
    slippage and commission costs. Assumes prices are normalized (e.g., set to 1.0),
    or can be overridden by injecting actual prices.
    """

    def __init__(self, slippage: float = 0.0001, commission_rate: float = 0.0001):
        self.slippage = slippage
        self.commission_rate = commission_rate

    @property
    def name(self) -> str:
        return "fixed"

    def estimate(self, trades: pd.Series, prices: pd.Series) -> pd.Series:
        """
        Estimate per-symbol transaction cost as a fraction of notional exposure.

        Args:
            trades: Difference between current and proposed weights
            prices: Price per symbol (defaults to 1.0 in practice)

        Returns:
            Estimated cost per symbol
        """
        rate = self.slippage + self.commission_rate
        return (trades.abs() * prices * rate).fillna(0.0)

    def adjust(self, signals: pd.Series, features: pd.DataFrame) -> pd.Series:
        """
        Adjust signals by subtracting estimated transaction costs.
        """
        # Use column from features or fallback to 0 portfolio
        current = features.get("current_position", pd.Series(0.0, index=signals.index))

        if isinstance(current, pd.DataFrame):
            current = current.squeeze()

        trades = signals.subtract(current, fill_value=0.0)

        dummy_prices = pd.Series(1.0, index=trades.index)
        estimated_cost = self.estimate(trades, dummy_prices)

        penalty_factor = 0.001
        adjusted = signals - (estimated_cost * penalty_factor)

        return adjusted
