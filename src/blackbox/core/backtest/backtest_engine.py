import traceback
from numbers import Number
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from blackbox.core.backtest.process_day import process_trading_day
from blackbox.core.types.context import PreparedDataBundle, StrategyContext
from blackbox.core.types.types import DailyLog


def run_backtest_loop(
    ctx: StrategyContext, data: PreparedDataBundle
) -> Tuple[List[DailyLog], List[dict]]:
    logs: List[DailyLog] = []
    trade_records: List[dict] = []
    previous_portfolio: dict[str, float] = {}

    matrix = data.feature_matrix
    warmup = data.metadata.warmup
    valid_dates = sorted(data.metadata.dates)
    snapshots = data.snapshots
    logger = ctx.logger

    logger.info(f"▶️ Starting backtest over {len(snapshots)} days")

    with logger.progress() as progress:
        task = progress.add_task("Backtesting", total=len(snapshots))

        for idx, snapshot in enumerate(snapshots):
            date = snapshot.date
            if date < data.metadata.min_date or date not in data.metadata.dates:
                progress.advance(task)
                continue

            try:
                try:
                    features_today = matrix.xs(date, level="date", drop_level=False)
                except KeyError:
                    logger.warning(
                        f"[{date.date()}] ⚠️ No features for this date — skipping"
                    )
                    progress.advance(task)
                    continue

                start_idx = max(0, idx - warmup)
                window_dates = [
                    d for d in valid_dates[start_idx : idx + 1] if d <= date
                ]
                features_window = matrix.loc[
                    matrix.index.get_level_values("date").isin(window_dates)
                ]

                log = process_trading_day(
                    snapshot, features_today, features_window, ctx
                )

                prev_equity = logs[-1].equity if logs else None
                log_backtest_day(logger, log, previous_equity=prev_equity)
                logs.append(log)

                _record_trades_for_day(log, previous_portfolio, trade_records)
                if log.portfolio is not None:
                    previous_portfolio = log.portfolio.to_dict()

            except Exception as e:
                logger.error(f"[{date.date()}] ❌ Exception during trading day: {e}")
                logger.debug(traceback.format_exc())

            progress.advance(task)

    logger.info(f"✅ Backtest complete — {len(logs)} trading days processed")
    inspect_equity_spikes(logs, logger)

    return logs, trade_records


def log_backtest_day(logger, log: DailyLog, previous_equity: float | None = None):
    date = log.date.date()

    def safe(x: float | None) -> float:
        return float(x) if isinstance(x, Number) else 0.0

    equity, cash, drawdown = map(safe, (log.equity, log.cash, log.drawdown))
    pnl = equity - previous_equity if previous_equity is not None else 0.0
    n_trades = log.trades.astype(bool).sum() if log.trades is not None else 0

    pnl_str = f"${pnl:,.2f}"
    if pnl > 50:
        pnl_str = f"🟢 +{pnl_str}"
    elif pnl < -50:
        pnl_str = f"🔴 -${abs(pnl):,.2f}"

    if n_trades > 0 or abs(pnl) > 1.0:
        logger.info(
            f"[{date}] 💰 Equity: ${equity:>10,.2f} | "
            f"💵 Cash: ${cash:>10,.2f} | "
            f"📉 Drawdown: {drawdown:6.2f}% | "
            f"🔄 Trades: {n_trades:>3d} | "
            f"💹 PnL: {pnl_str:>10}"
        )

        if log.ic is not None:
            logger.debug(f"[{date}] 🔗 Information Coefficient (IC): {log.ic:.4f}")

        if log.trades is not None and not log.trades.empty:
            top = log.trades.abs().nlargest(3).index.tolist()
            logger.debug(f"[{date}] 🚀 Top traded: {', '.join(top)}")


def _record_trades_for_day(
    log: DailyLog, previous_portfolio: dict, trade_records: List[dict]
):
    if log.trades is not None and not log.trades.empty:
        for symbol, weight in log.trades.items():
            if abs(weight) < 1e-8:
                continue
            prev_weight = previous_portfolio.get(symbol, 0.0)
            action = "adjust" if abs(prev_weight) >= 1e-8 else "enter"
            price = log.prices.get(symbol, float("nan"))
            notional = weight * price * log.equity if price and log.equity else 0.0
            trade_records.append(
                {
                    "date": log.date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "weight": float(weight),
                    "price": float(price),
                    "notional": float(notional),
                    "action": action,
                }
            )

    if log.portfolio is not None:
        exited = {
            sym: w
            for sym, w in previous_portfolio.items()
            if abs(w) > 1e-8 and sym not in log.portfolio.index
        }
        for symbol in exited:
            trade_records.append(
                {
                    "date": log.date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "weight": 0.0,
                    "price": float(log.prices.get(symbol, float("nan"))),
                    "notional": 0.0,
                    "action": "exit",
                }
            )


def inspect_equity_spikes(logs: list[DailyLog], logger, threshold: float = 30.0):
    prev_equity = None
    for log in logs:
        equity = float(log.equity or 0.0)
        date = log.date.date()
        pnl = equity - prev_equity if prev_equity is not None else 0.0

        if pnl > threshold:
            logger.warning(
                f"⚠️ Spike on {date} — PnL: ${pnl:.2f}, Equity: ${equity:.2f}"
            )
            logger.info(
                f"Cash: ${log.cash:.2f} | Trades: {log.trades.astype(bool).sum()}"
            )
            if log.trades is not None and not log.trades.empty:
                top = log.trades.abs().nlargest(5)
                for symbol in top.index:
                    logger.info(f" - {symbol}: weight={log.trades[symbol]:.4f}")
            else:
                logger.info("No trades recorded.")

        prev_equity = equity
