import pandas as pd

from trader.core.execution import ExecutionModel
from trader.core.slippage import SlippageModel


class SimpleExecutionModel(ExecutionModel):
    def __init__(self, slippage_model: SlippageModel | None = None):
        self.slippage_model = slippage_model

    def execute_orders(
        self, target_positions: pd.DataFrame, market_data: pd.DataFrame
    ) -> pd.DataFrame:
        fills = []
        for symbol, row in target_positions.iterrows():
            weight = row["target_weight"]
            symbol_data = market_data[market_data["symbol"] == symbol]
            if symbol_data.empty:
                continue

            price = symbol_data["close"].iloc[-1]
            side = "buy" if weight > 0 else "sell"

            fills.append(
                {
                    "symbol": symbol,
                    "price": price,
                    "side": side,
                    "quantity": abs(weight),
                    "close_price": price,
                }
            )

        fills_df = pd.DataFrame(fills).set_index("symbol")

        if self.slippage_model:
            fills_df = self.slippage_model.apply(fills_df, market_data)
        else:
            fills_df["fill_price"] = fills_df["price"]

        fills_df["slippage"] = fills_df["fill_price"] - fills_df["close_price"]
        fills_df["slippace_pct"] = (
            fills_df["slippage"] / fills_df["close_price"] * 10000  # basis points
        )
        return fills_df
