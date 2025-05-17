import pandas as pd

from blackbox.models.interfaces import TransactionCostModel


class LinearMarketImpact(TransactionCostModel):
    """
    Transaction cost model that applies a linear market impact penalty.

    Cost is proportional to the absolute trade size (|target - current|).
    """

    def __init__(self, impact_coefficient: float = 0.001):
        """
        Args:
            impact_coefficient: Cost per unit change in portfolio weight.
        """
        self.impact_coefficient = impact_coefficient

    @property
    def name(self) -> str:
        return "linear_market_impact"

    def estimate(self, trades: pd.Series) -> pd.Series:
        """
        Estimate per-symbol cost based on linear impact model.

        Args:
            trades: Difference between proposed and current weights

        Returns:
            Cost penalty per symbol
        """
        return (trades.abs() * self.impact_coefficient).fillna(0.0)

    def adjust(self, signals: pd.Series, features: pd.DataFrame) -> pd.Series:
        """
        Adjust target weights by penalizing large trades.

        Args:
            signals: Proposed target weights
            features: Feature matrix (must include 'current_position')

        Returns:
            Cost-adjusted weights
        """
        current = features.get("current_position", pd.Series(0.0, index=signals.index))
        trades = signals.subtract(current, fill_value=0.0)
        cost_penalty = self.estimate(trades)

        # Apply linear cost as a shrinkage penalty
        adjusted = signals * (1.0 - cost_penalty.clip(upper=1.0))

        return adjusted
