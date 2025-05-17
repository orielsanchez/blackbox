import pandas as pd

from blackbox.models.interfaces import TransactionCostModel


class QuadraticImpact(TransactionCostModel):
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

    @property
    def name(self) -> str:
        return "quadratic_market_impact"

    def adjust(self, signals: pd.Series, features: pd.DataFrame) -> pd.Series:
        """
        Adjusts signals based on estimated transaction costs.

        Assumes both `signals` and `current_position` are in weight (notional) space.
        """
        if "current_position" in features.columns:
            current = features["current_position"]
        else:
            current = pd.Series(0.0, index=signals.index)

        delta = signals.sub(current, fill_value=0.0).fillna(0.0)
        adjusted = signals.copy()

        cost_penalty = pd.Series(0.0, index=signals.index, dtype=float)

        for symbol, change in delta.items():
            notional = abs(change)
            commission = max(self.commission_rate * notional, self.min_commission)
            impact = self.impact_coefficient * notional**2
            total_cost = commission + impact
            cost_penalty[symbol] = total_cost

        # Apply cost as a shrinkage factor
        shrinkage = 1.0 - cost_penalty.clip(upper=1.0)
        adjusted *= shrinkage

        return adjusted
