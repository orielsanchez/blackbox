from typing import Dict, Optional

import pandas as pd

from blackbox.core.types.types import OHLCVSnapshot, PortfolioTarget, TradeResult
from blackbox.models.interfaces import ExecutionModel
from blackbox.utils.context import get_logger
from blackbox.utils.logger import RichLogger


class MarketExecution(ExecutionModel):
    """Slippage-aware market order execution tracking share-based positions."""

    def __init__(
        self,
        slippage: float = 0.0002,
        commission: float = 0.0001,
        initial_portfolio_value: float = 1_000_000.0,
        fractional: bool = True,
        allow_shorts: bool = True,
        min_notional: float = 1.0,
        logger: Optional[RichLogger] = None,
    ):
        self._slippage = slippage
        self._commission = commission
        self._fractional = fractional
        self._allow_shorts = allow_shorts
        self._min_notional = min_notional

        self._portfolio_value = initial_portfolio_value
        self._current_cash = initial_portfolio_value
        self._positions: pd.Series = pd.Series(dtype=float)

        self._history: list[tuple[pd.Series, Dict[str, Dict]]] = []
        self.logger = logger or get_logger()

    @property
    def name(self) -> str:
        return "market"

    @property
    def portfolio_value(self) -> float:
        return self._portfolio_value

    @property
    def current_cash(self) -> float:
        return self._current_cash

    def execute(
        self,
        target: PortfolioTarget,
        current: pd.Series,
        snapshot: OHLCVSnapshot,
        prices: pd.Series,
        open_prices: pd.Series,
        date: pd.Timestamp,
    ) -> TradeResult:
        """Execute trades, applying slippage and commissions."""
        trades = target.weights - current
        capital = target.capital
        new_positions = self._positions.copy()

        executed, fill_prices = pd.Series(dtype=float), pd.Series(dtype=float)
        feedback: dict[str, dict] = {}

        for symbol, weight_delta in trades.items():
            price = prices.get(symbol)
            if pd.isna(price) or price <= 0:
                self.logger.warning(
                    f"[Execution] Invalid price for {symbol}, skipping."
                )
                continue

            notional = weight_delta * capital
            shares_delta = notional / price
            if not self._fractional:
                shares_delta = round(shares_delta)

            new_position = new_positions.get(symbol, 0.0) + shares_delta
            trade_notional = abs(shares_delta * price)

            if not self._allow_shorts and new_position < 0:
                self.logger.debug(f"[Execution] Skipping short trade for {symbol}.")
                continue

            if trade_notional < self._min_notional:
                self.logger.debug(
                    f"[Execution] Trade notional ${trade_notional:.2f} for {symbol} below minimum."
                )
                continue

            slippage_cost = trade_notional * self._slippage
            commission_cost = max(trade_notional * self._commission, 0.01)
            total_cost = trade_notional + slippage_cost + commission_cost

            if self._current_cash < total_cost:
                self.logger.debug(
                    f"[Execution] Insufficient cash (${self._current_cash:.2f}) for trade {symbol} (${total_cost:.2f})."
                )
                continue

            # Execute trade
            new_positions[symbol] = new_position
            executed[symbol] = shares_delta
            fill_prices[symbol] = price * (
                1 + self._slippage * (1 if shares_delta > 0 else -1)
            )

            feedback[symbol] = {
                "fill_price": fill_prices[symbol],
                "slippage": self._slippage,
                "commission": commission_cost,
                "notional": notional,
                "cost": total_cost,
                "direction": "buy" if shares_delta > 0 else "sell",
            }

            self._current_cash -= total_cost

        self._positions = new_positions[abs(new_positions) > 1e-6].sort_index()
        self.record(executed.copy(), feedback.copy())

        if self._current_cash < 0:
            self.logger.warning("[Execution] Negative cash balance detected!")

        return TradeResult(
            executed=executed,
            fill_prices=fill_prices,
            feedback=feedback,
        )

    def record(self, trades: pd.Series, feedback: Dict) -> None:
        """Record executed trades and their feedback."""
        self._history.append((trades, feedback))

    def update_portfolio(
        self,
        current: pd.Series,
        trades: pd.Series,
        capital: float,
        prices: pd.Series,
    ) -> pd.Series:
        """Update internal portfolio positions after trades."""
        positions = current.copy()
        for symbol, trade_shares in trades.items():
            positions[symbol] = positions.get(symbol, 0.0) + trade_shares

        positions = positions[abs(positions) > 1e-6].sort_index()
        self._positions = positions

        invested = (positions * prices).sum(min_count=1)
        self._current_cash = capital - invested
        self._portfolio_value = invested + self._current_cash

        return positions

    def mark_to_market(self, prices: pd.Series) -> None:
        """Update portfolio value based on current market prices."""
        positions_value = (self._positions * prices).sum(min_count=1)

        missing_prices = self._positions.index[
            ~self._positions.index.isin(prices.index)
        ]
        for symbol in missing_prices:
            self.logger.warning(f"[MTM] Missing price for {symbol}")

        self._portfolio_value = self._current_cash + positions_value
        self.logger.debug(
            f"[MTM] Portfolio=${self._portfolio_value:.2f}, Cash=${self._current_cash:.2f}"
        )

    def get_available_capital(self) -> float:
        """Return current portfolio valuation."""
        return self._portfolio_value

    def __repr__(self) -> str:
        return (
            f"<MarketExecution portfolio=${self._portfolio_value:,.2f}, "
            f"cash=${self._current_cash:,.2f}, positions={len(self._positions)}>"
        )
