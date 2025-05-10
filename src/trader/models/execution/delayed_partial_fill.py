import numpy as np
import pandas as pd

from trader.core.execution import ExecutionModel


class DelayedPartialFillExecutionModel(ExecutionModel):
    def __init__(self, delay_prob: float = 0.1, fill_rate_mean: float = 0.8):
        self.delay_prob = delay_prob
        self.fill_rate_mean = fill_rate_mean

    def execute(
        self,
        orders: pd.DataFrame,
        snapshot: pd.DataFrame,
        timestamp: pd.Timestamp,
        cash: float,
    ) -> list[dict]:
        fills = []
        prices = snapshot.set_index("symbol")["price"].to_dict()

        for _, order in orders.iterrows():
            symbol = order["symbol"]
            side = order["side"]
            qty = order["shares"]

            if np.random.rand() < self.delay_prob:
                continue  # simulate delay

            fill_rate = np.clip(np.random.normal(self.fill_rate_mean, 0.1), 0.1, 1.0)
            filled_qty = int(qty * fill_rate)
            price = prices.get(symbol, np.nan)
            if pd.isna(price) or filled_qty == 0:
                continue

            fills.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "quantity": filled_qty,
                    "fill_price": price,
                    "slippage": 0.0,  # slippage is modeled separately
                }
            )

        return fills
