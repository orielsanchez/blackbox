from typing import Optional

import pandas as pd

from trader.core.execution import ExecutionModel
from trader.core.slippage import SlippageModel


class SimpleExecutionModel(ExecutionModel):
    def __init__(self, slippage_model: Optional[SlippageModel] = None):
        self.slippage_model = slippage_model
        self.positions = {}  # symbol -> current position

    def execute(
        self,
        orders: pd.DataFrame,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        cash: float,
    ) -> list[dict]:
        required_order_cols = {"symbol", "shares", "side"}
        required_md_cols = {"symbol", "price"}

        if not required_order_cols.issubset(orders.columns):
            raise ValueError(
                f"Orders missing columns: {required_order_cols - set(orders.columns)}"
            )

        if not required_md_cols.issubset(market_data.columns):
            raise ValueError(
                f"Market data missing columns: {required_md_cols - set(market_data.columns)}"
            )

        orders = orders.copy()
        market_data = market_data.copy()
        orders["symbol"] = orders["symbol"].astype(str).str.upper()
        market_data["symbol"] = market_data["symbol"].astype(str).str.upper()
        market = market_data.set_index("symbol")

        # Compute slippage if needed
        if self.slippage_model is not None:
            orders["price"] = orders["symbol"].map(market["price"])
            slippage_df = self.slippage_model.score(orders, timestamp)
            slippage_dict = dict(zip(slippage_df["symbol"], slippage_df["slippage"]))
        else:
            slippage_dict = {}

        fills = []
        cash_remaining = cash

        for _, order in orders.iterrows():
            symbol = order["symbol"]
            side = order["side"].lower()
            quantity = int(order["shares"])

            if quantity <= 0 or symbol not in market.index:
                continue

            price = market.at[symbol, "price"]
            if pd.isna(price):
                continue

            slippage = slippage_dict.get(symbol, round(price * 0.001, 6))
            fill_price = (
                round(price + slippage, 6)
                if side == "buy"
                else round(price - slippage, 6)
            )

            if side == "buy":
                total_cost = quantity * fill_price
                if total_cost > cash_remaining:
                    quantity = int(cash_remaining // fill_price)
                    if quantity <= 0:
                        continue
                    total_cost = quantity * fill_price
                cash_remaining -= total_cost
            else:
                total_cost = quantity * fill_price  # sell earns money

            previous_position = self.positions.get(symbol, 0)
            new_position = (
                previous_position + quantity
                if side == "buy"
                else previous_position - quantity
            )
            self.positions[symbol] = new_position

            fills.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "fill_price": fill_price,
                    "slippage": slippage,
                    "position_after": new_position,
                    "notional": fill_price * quantity,
                }
            )

        return fills
