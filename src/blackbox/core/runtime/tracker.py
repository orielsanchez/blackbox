from typing import Dict, List, Optional, Set

import pandas as pd


class PositionMeta:
    """
    Metadata for a position, including acquisition details and holding period.
    """

    def __init__(
        self,
        entry_date: pd.Timestamp,
        entry_price: float,
        quantity: float,
        cost_basis: float = 0.0,
    ):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.quantity = quantity
        self.cost_basis = cost_basis
        self.last_updated = entry_date
        self.holding_days = 0
        self.last_price = entry_price

    def update(
        self, date: pd.Timestamp, price: float, quantity: float, cost: float = 0.0
    ):
        days_since_last = (date - self.last_updated).days
        self.holding_days += max(0, days_since_last)
        self.last_updated = date
        self.last_price = price

        if quantity != 0:
            total_quantity = self.quantity + quantity

            if total_quantity == 0:
                self.entry_price = 0.0
                self.cost_basis = 0.0
            else:
                if quantity > 0:
                    total_value = (self.entry_price * self.quantity) + (
                        price * quantity
                    )
                    self.entry_price = total_value / max(total_quantity, 1e-8)
                    self.cost_basis += cost
                else:
                    if self.quantity != 0:
                        reduction_ratio = abs(quantity) / max(abs(self.quantity), 1e-8)
                        reduction_ratio = min(reduction_ratio, 1.0)
                        self.cost_basis *= 1 - reduction_ratio
            self.quantity = total_quantity


class PositionTracker:
    """
    Enhanced position tracker that supports pending orders and realistic execution.
    """

    def __init__(self, initial_cash: float = 1000000.0):
        self.initial_cash = initial_cash
        self.current_cash = initial_cash
        self.positions: Dict[str, PositionMeta] = {}
        self.pending_orders: Optional[pd.DataFrame] = None
        self.trade_history: List[Dict] = []
        self.min_holding_period = 0

    def set_min_holding_period(self, days: int):
        self.min_holding_period = days

    def get_portfolio(self) -> pd.Series:
        return pd.Series(
            {s: p.quantity for s, p in self.positions.items() if p.quantity != 0}
        )

    def record_pending_orders(self, orders: pd.DataFrame):
        self.pending_orders = orders.copy()

    def get_pending_orders(self) -> Optional[pd.DataFrame]:
        return self.pending_orders

    def clear_pending_orders(self):
        self.pending_orders = None

    def compute_portfolio_value(self, prices: pd.Series) -> float:
        equity = self.current_cash
        for symbol, pos in self.positions.items():
            if pos.quantity > 0 and symbol in prices:
                equity += pos.quantity * prices[symbol]
        return equity

    def update(self, trades: pd.Series, prices: pd.Series):
        now = pd.Timestamp.now()
        for symbol, quantity in trades.items():
            if quantity == 0 or symbol not in prices:
                continue

            price = prices[symbol]
            trade_value = abs(quantity * price)
            cost = trade_value * 0.0005

            self.current_cash -= quantity * price + cost

            if symbol in self.positions:
                self.positions[symbol].update(now, price, quantity, cost)
                if self.positions[symbol].quantity == 0:
                    del self.positions[symbol]
            else:
                self.positions[symbol] = PositionMeta(now, price, quantity, cost)

        self.clear_pending_orders()

    def get_tradable_symbols(self, current_date: pd.Timestamp) -> Set[str]:
        return {
            s
            for s, p in self.positions.items()
            if (current_date - p.entry_date).days >= self.min_holding_period
        }

    def get_available_capital(self) -> float:
        return self.current_cash
