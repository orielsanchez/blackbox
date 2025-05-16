from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class TradeResult:
    executed: pd.Series
    fill_prices: pd.Series
    feedback: Dict[str, Dict]


@dataclass
class DailyLog:
    date: pd.Timestamp
    prices: pd.Series
    trades: pd.Series
    portfolio: pd.Series
    feedback: Dict[str, Any]
    equity: float
    cash: float


def reconcile_trades(current: pd.Series, target: pd.Series) -> pd.Series:
    """Compute the required trade weights to move from current to target."""
    all_symbols = current.index.union(target.index)
    delta = target.reindex(all_symbols, fill_value=0.0) - current.reindex(
        all_symbols, fill_value=0.0
    )
    return delta[delta.abs() > 1e-6]


def simulate_execution(
    trades: pd.Series, prices: pd.Series, slippage: float, capital: float
) -> TradeResult:
    """Simulate slippage-adjusted execution of trades with cost feedback."""
    valid_trades = trades[trades.index.isin(prices.index)]
    executed = valid_trades.copy()
    fill_prices = pd.Series(index=executed.index, dtype=float)
    feedback = {}

    for symbol, weight in executed.items():
        raw_price = prices[symbol]
        direction = 1 if weight > 0 else -1
        slip_pct = slippage
        fill_price = raw_price * (1 + slip_pct * direction)

        notional = weight * capital
        trade_cost = abs(notional * slip_pct)

        fill_prices[symbol] = fill_price
        feedback[symbol] = {
            "fill_price": fill_price,
            "slippage": slip_pct,
            "notional": notional,
            "cost": trade_cost,
            "direction": "buy" if direction > 0 else "sell",
        }

    return TradeResult(executed=executed, fill_prices=fill_prices, feedback=feedback)
