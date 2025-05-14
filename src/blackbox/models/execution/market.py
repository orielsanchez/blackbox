from typing import Dict

import pandas as pd

from blackbox.models.interfaces import ExecutionModel
from blackbox.utils.context import get_logger


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
        self.current_cash = initial_portfolio_value
        self.positions = pd.Series(dtype=float)  # symbol -> weight
        self.history = []
        self.logger = get_logger()

    def record(self, trades: pd.Series, feedback: Dict[str, Dict]):
        self.history.append((trades.copy(), feedback.copy()))

    def update_portfolio(self, current: pd.Series, trades: pd.Series) -> pd.Series:
        """
        Applies trade deltas in terms of weights.
        Updates internal weights and cash.
        """
        new_weights = current.copy()

        for symbol, weight_delta in trades.items():
            if not self.fractional:
                weight_delta = round(weight_delta, 4)

            proposed_weight = new_weights.get(symbol, 0.0) + weight_delta
            notional = abs(weight_delta * self.portfolio_value)

            if not self.allow_shorts and proposed_weight < 0:
                self.logger.debug(f"[Execution] Skipping short trade: {symbol}")
                continue

            if notional < self.min_notional:
                self.logger.debug(
                    f"[Execution] Skipping {symbol} — notional ${notional:.2f} < min ${self.min_notional}"
                )
                continue

            new_weights[symbol] = proposed_weight
            self.logger.debug(
                f"[Execution] {symbol}: Δweight={weight_delta:.6f} → new_weight={proposed_weight:.6f}"
            )

        self.positions = new_weights.copy()
        self.current_cash = self.portfolio_value * (1.0 - new_weights.abs().sum())

        return self.positions[self.positions.abs() > 1e-6].sort_index()

    def get_available_capital(self) -> float:
        """Returns the total portfolio capital (used for weighting)."""
        return self.portfolio_value

    def mark_to_market(self, prices: pd.Series):
        """Recalculate portfolio value based on current prices and weights."""
        total_value = 0.0
        for symbol, weight in self.positions.items():
            if symbol in prices:
                total_value += weight * prices[symbol]
            else:
                self.logger.warning(f"[MTM] Missing price for {symbol}, skipping.")

        self.portfolio_value = total_value + self.current_cash
        self.logger.debug(
            f"[MTM] Portfolio value updated: ${self.portfolio_value:,.2f} (cash: ${self.current_cash:,.2f})"
        )
