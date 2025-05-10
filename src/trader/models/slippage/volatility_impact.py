import pandas as pd

from trader.core.slippage import SlippageModel


class VolatilityImpactSlippageModel(SlippageModel):
    def __init__(self, impact_factor: float = 0.1):
        self.impact_factor = impact_factor
        self.rolling_window = 10

    def score(self, positions: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        if not {"symbol", "shares", "price"}.issubset(positions.columns):
            raise ValueError("Missing required columns in positions")

        # You can optionally cache or pass in volatility externally for efficiency
        positions = positions.copy()
        volatility = 0.02  # Placeholder if no real data available
        positions["slippage"] = self.impact_factor * volatility * positions["price"]
        return positions[["symbol", "slippage"]]
