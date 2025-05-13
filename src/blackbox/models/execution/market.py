from typing import Dict

import pandas as pd

from blackbox.models.interfaces import ExecutionModel


class MarketExecution(ExecutionModel):
    name = "market"

    def __init__(
        self,
        slippage: float = 0.0002,
        commission: float = 0.0001,
        initial_portfolio_value: float = 1_000_000.0,
        fractional: bool = True,
        allow_shorts: bool = True,
        min_notional: float = 1.0,
    ):
        self.slippage = slippage
        self.commission = commission
        self.fractional = fractional
        self.allow_shorts = allow_shorts
        self.min_notional = min_notional
        self.portfolio_value = initial_portfolio_value
        self.history = []

    def record(self, trades: pd.Series, feedback: Dict[str, Dict]):
        """Store trade + cost info for diagnostics or logging."""
        self.history.append((trades.copy(), feedback.copy()))

    def update_portfolio(self, current: pd.Series, trades: pd.Series) -> pd.Series:
        """Apply slippage-adjusted trades and update portfolio weights."""

        new_portfolio = current.copy()

        for symbol, weight_delta in trades.items():
            if not self.fractional:
                weight_delta = round(weight_delta, 4)

            notional = abs(weight_delta) * self.portfolio_value

            if (
                not self.allow_shorts
                and (new_portfolio.get(symbol, 0) + weight_delta) < 0
            ):
                continue  # skip disallowed short

            if notional < self.min_notional:
                continue  # skip dust trades

            new_portfolio[symbol] = new_portfolio.get(symbol, 0.0) + weight_delta

        # Normalize and return
        return new_portfolio[abs(new_portfolio) > 1e-6].sort_index()
