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
        initial_portfolio_value: float = None,
        fractional: bool = True,
        allow_shorts: bool = True,
        min_notional: float = 1.0,
    ):
        self.slippage = slippage
        self.commission = commission
        self.fractional = fractional
        self.allow_shorts = allow_shorts
        self.min_notional = min_notional

        self.portfolio_value = 0.0
        self.current_cash = initial_portfolio_value or 0.0

        self.initial_portfolio_value = initial_portfolio_value or 1_000_000
        self.portfolio_value = self.initial_portfolio_value
        self.current_cash = self.initial_portfolio_value
        self.positions = pd.Series(dtype=float)  # symbol -> weight
        self.history = []

        self.logger = get_logger()

    def record(self, trades: pd.Series, feedback: Dict[str, Dict]):
        self.history.append((trades.copy(), feedback.copy()))

    def update_portfolio(self, current: pd.Series, trades: pd.Series, capital: float) -> pd.Series:
        new_weights = current.copy()
        executed_trades = {}
        total_trade_cost = 0.0

        for symbol, weight_delta in trades.items():
            if not self.fractional:
                weight_delta = round(weight_delta, 4)

            prev_weight = new_weights.get(symbol, 0.0)
            new_weight = prev_weight + weight_delta
            notional = abs(weight_delta * capital)

            if not self.allow_shorts and new_weight < 0:
                self.logger.debug(f"[Execution] Skipping short trade: {symbol}")
                continue

            if notional < self.min_notional:
                self.logger.debug(
                    f"[Execution] Skipping {symbol} — notional ${notional:.2f} < min ${self.min_notional:.2f}"
                )
                continue

            slippage_cost = notional * self.slippage
            commission_cost = max(notional * self.commission, 0.01)
            total_cost = notional + slippage_cost + commission_cost

            if self.current_cash - total_trade_cost < total_cost:
                self.logger.debug(
                    f"[Execution] Skipping {symbol} — cost ${total_cost:.2f} exceeds available cash ${self.current_cash - total_trade_cost:.2f}"
                )
                continue

            # Accept trade
            new_weights[symbol] = new_weight
            executed_trades[symbol] = weight_delta
            total_trade_cost += total_cost

            self.logger.debug(
                f"[Execution] {symbol}: Δweight={weight_delta:.6f} → new_weight={new_weight:.6f}, cost=${total_cost:.2f}"
            )

        self.positions = new_weights.copy()
        self.current_cash -= total_trade_cost

        if self.current_cash <= 0:
            self.logger.warning(
                "[Execution] 🚨 Cash depleted or zero! Execution may behave incorrectly."
            )

        return self.positions[self.positions.abs() > 1e-6].sort_index()

    def get_available_capital(self) -> float:
        return self.portfolio_value

    def mark_to_market(self, prices: pd.Series):
        weighted_values = {
            symbol: weight * prices[symbol]
            for symbol, weight in self.positions.items()
            if symbol in prices
        }

        missing = [s for s in self.positions.index if s not in prices]
        for symbol in missing:
            self.logger.warning(f"[MTM] Missing price for {symbol}, skipping.")

        position_value = sum(weighted_values.values())
        self.portfolio_value = position_value + self.current_cash

        self.logger.debug(
            f"[MTM] Portfolio value updated: ${self.portfolio_value:,.2f} "
            f"(cash: ${self.current_cash:,.2f})"
        )

    def __repr__(self):
        return (
            f"<MarketExecution portfolio=${self.portfolio_value:,.2f}, "
            f"cash=${self.current_cash:,.2f}, positions={len(self.positions)}>"
        )
