import pandas as pd

from blackbox.models.interfaces import RiskModel


class PositionLimitRisk(RiskModel):
    name = "position_limit"

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
        if signals.empty:
            return pd.Series(dtype=float)

        # Step 1: Rank signals by absolute value
        ranked = signals.abs().sort_values(ascending=False)

        # Step 2: Determine how many positions we can afford
        max_positions = int(self.max_leverage / self.max_position_size)
        selected_symbols = ranked.head(max_positions).index

        # Step 3: Take only top signals and clip them
        selected = signals[selected_symbols]
        selected = selected.clip(
            lower=-self.max_position_size if self.allow_shorts else 0.0,
            upper=self.max_position_size,
        )

        # Step 4: Normalize weights to match total leverage (optional)
        total_abs = selected.abs().sum()
        if total_abs > 0:
            selected *= self.max_leverage / total_abs

        return selected
