import pandas as pd

from blackbox.models.interfaces import RiskModel


class PositionLimitRisk(RiskModel):
    name = "position_limit"
    """
    Risk model that clips weights based on:
    - max position size
    - optional short-selling ban
    - optional leverage normalization
    """

    def __init__(
        self,
        max_position_size: float = 0.1,
        max_leverage: float = 1.0,
        allow_shorts: bool = True,
    ):
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.allow_shorts = allow_shorts

    def apply(self, signals: pd.Series, current_portfolio: pd.Series) -> pd.Series:
        weights = signals.copy()

        # Clip weights
        weights = weights.clip(
            lower=-self.max_position_size if self.allow_shorts else 0.0,
            upper=self.max_position_size,
        )

        # Normalize to total leverage if needed
        total_abs = weights.abs().sum()
        if total_abs > self.max_leverage and total_abs > 0:
            weights *= self.max_leverage / total_abs

        return weights
