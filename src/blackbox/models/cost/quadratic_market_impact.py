import pandas as pd

from blackbox.models.interfaces import TransactionCostModel


class QuadraticImpact(TransactionCostModel):
    name = "quadratic_market_impact"
    """
    Transaction cost model that applies:
    - fixed commission on notional
    - quadratic market impact (notional²-based cost)
    """

    def __init__(
        self,
        commission_rate: float = 0.0001,  # 1 basis point
        impact_coefficient: float = 0.00005,  # scales quadratic penalty
        min_commission: float = 0.0,  # optional minimum commission
    ):
        self.commission_rate = commission_rate
        self.impact_coefficient = impact_coefficient
        self.min_commission = min_commission

    def adjust(self, target: pd.Series, current: pd.Series) -> pd.Series:
        """
        Shrink weights based on estimated costs from moving between current and target.

        Args:
            target (pd.Series): Proposed target weights
            current (pd.Series): Current portfolio weights

        Returns:
            pd.Series: Adjusted target weights
        """
        delta = target.sub(current, fill_value=0.0)

        adjusted = target.copy()

        for symbol, weight_change in delta.items():
            notional = abs(weight_change)

            # Cost = linear + quadratic impact
            commission = max(self.commission_rate * notional, self.min_commission)
            impact = self.impact_coefficient * (notional**2)
            total_cost = commission + impact

            # Reduce the proposed size to reflect the cost penalty
            adjusted[symbol] = adjusted.get(symbol, 0.0) - total_cost * (
                1 if weight_change > 0 else -1
            )

        return adjusted
