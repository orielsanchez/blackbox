from typing import Dict, Optional

import numpy as np
import pandas as pd

from blackbox.core.types.types import OHLCVSnapshot, PortfolioTarget, TradeResult
from blackbox.models.interfaces import ExecutionModel
from blackbox.utils.context import get_logger
from blackbox.utils.logger import RichLogger


class MarketExecution(ExecutionModel):
    """Enhanced slippage-aware market order execution with limit and VWAP simulation."""

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
        self._entry_prices: Dict[str, float] = {}  # for stop-loss logic

        self._history: list[tuple[pd.Series, Dict[str, Dict]]] = []
        self.logger = logger or get_logger()

    @property
    def name(self) -> str:
        return "market2"

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
        """Execute trades using specified method: market, limit, vwap."""
        trades = target.weights - current
        capital = target.capital
        new_positions = self._positions.copy()

        executed, fill_prices = pd.Series(dtype=float), pd.Series(dtype=float)
        feedback: dict[str, dict] = {}

        # Optional fields from PortfolioTarget
        execution_method = getattr(target, "execution_method", "market")
        limit_prices: Optional[pd.Series] = getattr(target, "limit_prices", None)

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

            # ─── Simulated execution logic ───
            fill_price = price
            stop_triggered = False

            if execution_method == "limit":
                limit_price = (
                    limit_prices.get(symbol) if limit_prices is not None else None
                )
                if limit_price is None:
                    self.logger.debug(
                        f"[Execution] No limit price for {symbol}, skipping."
                    )
                    continue
                if (shares_delta > 0 and price > limit_price) or (
                    shares_delta < 0 and price < limit_price
                ):
                    self.logger.debug(
                        f"[Execution] Limit not satisfied for {symbol}, skipping."
                    )
                    continue
                fill_price = limit_price

            elif execution_method == "vwap":
                fill_price = self.estimate_vwap(symbol, price)

            else:  # market
                fill_price = price * (
                    1 + self._slippage * (1 if shares_delta > 0 else -1)
                )

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
            fill_prices[symbol] = fill_price
            self._entry_prices[symbol] = fill_price  # record for stop-loss use
            self._current_cash -= total_cost

            feedback[symbol] = {
                "fill_price": fill_price,
                "slippage": self._slippage,
                "commission": commission_cost,
                "notional": notional,
                "cost": total_cost,
                "direction": "buy" if shares_delta > 0 else "sell",
                "execution_type": execution_method,
                "limit_price": (
                    limit_prices.get(symbol) if limit_prices is not None else None
                ),
                "stop_loss_triggered": stop_triggered,
            }

        self._positions = new_positions[abs(new_positions) > 1e-6].sort_index()
        self.record(executed.copy(), feedback.copy())

        if self._current_cash < 0:
            self.logger.warning("[Execution] Negative cash balance detected!")

        return TradeResult(
            executed=executed,
            fill_prices=fill_prices,
            feedback=feedback,
        )

    def estimate_vwap(self, symbol: str, reference_price: float) -> float:
        """Simulate a slippage-adjusted VWAP fill price."""
        noise = np.random.normal(0, 0.0003)
        return reference_price * (1 + noise)

    def check_stop_loss(
        self, prices: pd.Series, atr: pd.Series, k: float = 2.0
    ) -> pd.Series:
        """Return a signal vector to flatten positions that breach stop-loss."""
        to_close = []
        for symbol, entry_price in self._entry_prices.items():
            current_price = prices.get(symbol)
            atr_val = atr.get(symbol)
            if pd.isna(current_price) or pd.isna(atr_val):
                continue
            stop_price = (
                entry_price - k * atr_val
                if self._positions[symbol] > 0
                else entry_price + k * atr_val
            )
            if (self._positions[symbol] > 0 and current_price < stop_price) or (
                self._positions[symbol] < 0 and current_price > stop_price
            ):
                to_close.append(symbol)
                self.logger.info(
                    f"[StopLoss] Triggered for {symbol} at {current_price:.2f}"
                )
        return pd.Series(0.0, index=to_close)

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
        return self._portfolio_value

    def __repr__(self) -> str:
        return (
            f"<MarketExecution portfolio=${self._portfolio_value:,.2f}, "
            f"cash=${self._current_cash:,.2f}, positions={len(self._positions)}>"
        )
