import pandas as pd

from trader.core.slippage import SlippageModel


class PercentSlippageModel(SlippageModel):
    def __init__(self, slippage_rate: float = 0.001):
        self.slippage_rate = slippage_rate

    def apply(self, orders: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        adjusted = orders.copy()
        for i, row in adjusted.iterrows():
            side = row.get("side", "buy").lower()
            price = row["price"]
            slip = price * self.slippage_rate
            adjusted.at[i, "fill_price"] = (
                price + slip if side == "buy" else price - slip
            )
        return adjusted
