import pandas as pd

from trader.core.tx_cost import TxCostModel


class VolumeAdjustedTxCostModel(TxCostModel):
    def __init__(self, base_cost: float = 0.005, volume_impact_factor: float = 1e-5):
        self.base_cost = base_cost
        self.volume_impact_factor = volume_impact_factor

    def score(self, positions: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        if not {"symbol", "shares", "price"}.issubset(positions.columns):
            raise ValueError("Missing required columns in positions")

        positions = positions.copy()
        positions["notional"] = positions["shares"] * positions["price"]
        positions["tx_cost"] = (
            self.base_cost + self.volume_impact_factor * positions["notional"]
        )
        return positions[["symbol", "tx_cost"]]
