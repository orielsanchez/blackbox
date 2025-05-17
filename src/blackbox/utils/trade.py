from typing import Optional

import pandas as pd

from blackbox.core.types.types import TradeResult
from blackbox.utils.logger import RichLogger


def reconcile_trades(
    current: pd.Series,
    target: pd.Series,
    min_trade_size: float = 0.0001,
    max_position_size: float = 0.20,
) -> pd.Series:
    """
    Compute trade weights needed to transition from current to target portfolio.
    """
    symbols = current.index.union(target.index)
    current_aligned = pd.Series(0.0, index=symbols, dtype=float)
    target_aligned = pd.Series(0.0, index=symbols, dtype=float)

    current_aligned.loc[current.index] = current.astype(float)
    target_aligned.loc[target.index] = target.astype(float)

    target_capped = target_aligned.clip(-max_position_size, max_position_size)
    raw_trades = target_capped - current_aligned
    trades = raw_trades.where(raw_trades.abs() >= min_trade_size, 0.0)

    return trades.round(6)


def simulate_execution(
    trades: pd.Series,
    prices: pd.Series,
    decision_prices: Optional[pd.Series] = None,
    slippage: float = 0.001,
    capital: float = 1_000_000.0,
    logger: Optional[RichLogger] = None,
    date: Optional[pd.Timestamp] = None,
    max_adverse_price_pct: float = 0.03,
    min_execution_capital: float = 100.0,
) -> TradeResult:
    """
    Simulate execution of trades, applying price checks, slippage, and capital limits.
    """
    if logger:
        logger.debug(
            f"{date.date() if date else 'Unknown'} | Simulating execution for {len(trades)} trades"
        )

    if trades.empty:
        if logger:
            logger.debug("No trades to execute.")
        return TradeResult(
            executed=pd.Series(dtype=float),
            fill_prices=pd.Series(dtype=float),
            feedback={},
        )

    # Ensure prices exist
    missing = trades.index.difference(prices.index)
    if not missing.empty:
        if logger and 0:
            logger.warning(f"Missing execution prices for: {missing.tolist()}")
        trades = trades.drop(missing)
        if trades.empty:
            return TradeResult(
                executed=pd.Series(dtype=float),
                fill_prices=pd.Series(dtype=float),
                feedback={"errors": "All trades had missing prices"},
            )

    signs = trades.apply(lambda x: -1 if x < 0 else 1)
    adjusted_prices = prices.loc[trades.index] * (1 + signs * slippage)

    executed = pd.Series(0.0, index=trades.index, dtype=float)
    fill_prices = pd.Series(0.0, index=trades.index, dtype=float)
    feedback: dict[str, dict] = {}
    dollar_values = {}

    for symbol in trades.index:
        if decision_prices is not None:
            dp = decision_prices.get(symbol)
            ep = prices.get(symbol)
            if pd.notna(dp) and pd.notna(ep):
                change = (ep / dp) - 1
                if (trades[symbol] > 0 and change > max_adverse_price_pct) or (
                    trades[symbol] < 0 and change < -max_adverse_price_pct
                ):
                    feedback[symbol] = {
                        "status": "rejected",
                        "reason": "price_limit_exceeded",
                        "decision_price": dp,
                        "execution_price": ep,
                        "price_change_pct": change,
                    }
                    continue

        value = abs(trades[symbol] * adjusted_prices[symbol] * capital)
        dollar_values[symbol] = value
        if value < min_execution_capital:
            feedback[symbol] = {
                "status": "rejected",
                "reason": "trade_too_small",
                "dollar_value": value,
                "minimum": min_execution_capital,
            }

    remaining_capital = capital

    for symbol in trades.index:
        if symbol in feedback:
            continue

        if trades[symbol] < 0:
            # sell
            executed[symbol] = trades[symbol]
            fill_prices[symbol] = adjusted_prices[symbol]
            remaining_capital += dollar_values[symbol]
            feedback[symbol] = {
                "status": "filled",
                "fill_price": fill_prices[symbol],
                "original_price": prices[symbol],
                "dollar_value": dollar_values[symbol],
            }

    # buy orders
    buys = [s for s in trades.index if trades[s] > 0 and s not in feedback]
    buys.sort(key=lambda s: abs(trades[s]), reverse=True)

    for symbol in buys:
        cost = dollar_values[symbol]
        if cost > remaining_capital:
            scale = remaining_capital / cost
            executed[symbol] = trades[symbol] * scale
            fill_prices[symbol] = adjusted_prices[symbol]
            feedback[symbol] = {
                "status": "partial_fill",
                "fill_price": fill_prices[symbol],
                "original_price": prices[symbol],
                "executed_quantity": executed[symbol],
                "scaling_factor": scale,
                "requested_value": cost,
                "dollar_value": cost * scale,
            }
            break
        else:
            executed[symbol] = trades[symbol]
            fill_prices[symbol] = adjusted_prices[symbol]
            remaining_capital -= cost
            feedback[symbol] = {
                "status": "filled",
                "fill_price": fill_prices[symbol],
                "original_price": prices[symbol],
                "dollar_value": cost,
            }

    executed = executed[executed != 0]
    fill_prices = fill_prices.loc[executed.index]

    if logger:
        filled = sum(
            v["status"] in ("filled", "partial_fill") for v in feedback.values()
        )
        rejected = sum(v["status"] == "rejected" for v in feedback.values())
        logger.debug(
            f"{date.date() if date else 'Unknown'} | Execution summary: {filled} filled, {rejected} rejected, "
            f"${capital - remaining_capital:.2f} deployed"
        )

    return TradeResult(
        executed=executed,
        fill_prices=fill_prices,
        feedback=feedback,
    )
