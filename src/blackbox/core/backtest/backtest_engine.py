import traceback
from pathlib import Path
from typing import List

import pandas as pd

from blackbox.core.backtest.process_day import process_trading_day
from blackbox.core.types.context import PreparedDataBundle, StrategyContext
from blackbox.core.types.types import DailyLog


def log_backtest_day(logger, log: DailyLog, previous_equity: float | None = None):
    from numbers import Number

    date = log.date.date()

    def safe_float(x, default=0.0):
        return float(x) if isinstance(x, Number) else default

    equity = safe_float(log.equity)
    cash = safe_float(log.cash)
    drawdown = safe_float(log.drawdown)
    pnl = equity - previous_equity if previous_equity is not None else 0.0
    delta_trades = log.trades.astype(bool).sum() if log.trades is not None else 0

    drawdown_pct_str = f"{drawdown:6.2f}%" if drawdown else "   0.00%"
    pnl_str = f"${pnl:,.2f}"
    if pnl > 50:
        pnl_str = f"🟢 +{pnl_str}"
    elif pnl < -50:
        pnl_str = f"🔴 -${abs(pnl):,.2f}"

    if delta_trades > 0 or abs(pnl) > 1.0:
        logger.info(
            f"[{date}] "
            f"💰 Equity: ${equity:>10,.2f} | "
            f"💵 Cash: ${cash:>10,.2f} | "
            f"📉 Drawdown: {drawdown_pct_str} | "
            f"🔄 Trades: {delta_trades:>3d} | "
            f"💹 PnL: {pnl_str:>10}"
        )

        if log.trades is not None and not log.trades.empty:
            top_traded = log.trades.abs().sort_values(ascending=False).head(3).index.tolist()
            logger.debug(f"[{date}] 🚀 Top traded: {', '.join(top_traded)}")


def run_backtest_loop(
    ctx: StrategyContext, data: PreparedDataBundle, output_dir: Path
) -> List[DailyLog]:
    logs: list[DailyLog] = []
    trade_records: list[dict] = []
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
                    logger.warning(f"[{date.date()}] ⚠️ No features for this date — skipping")
                    progress.advance(task)
                    continue

                start_idx = max(0, idx - warmup)
                window_dates = [d for d in valid_dates[start_idx : idx + 1] if d <= date]
                features_window = matrix.loc[
                    matrix.index.get_level_values("date").isin(window_dates)
                ]

                log = process_trading_day(
                    snapshot=snapshot,
                    features_today=features_today,
                    features_window=features_window,
                    ctx=ctx,
                )

                prev_equity = logs[-1].equity if logs else None
                log_backtest_day(logger, log, previous_equity=prev_equity)
                logs.append(log)

                # Record new/adjust trades
                if log.trades is not None and not log.trades.empty:
                    for symbol, weight in log.trades.items():
                        if abs(weight) < 1e-8:
                            continue

                        prev_weight = previous_portfolio.get(symbol, 0.0)
                        action = "adjust"
                        if abs(prev_weight) < 1e-8:
                            action = "enter"

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

                # Record full exits
                if log.portfolio is not None:
                    exited_symbols = {
                        sym: prev_weight
                        for sym, prev_weight in previous_portfolio.items()
                        if abs(prev_weight) > 1e-8 and sym not in log.portfolio.index
                    }
                    for symbol, old_weight in exited_symbols.items():
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

                # Update state for next day
                if log.portfolio is not None:
                    previous_portfolio = log.portfolio.to_dict()

            except Exception as e:
                logger.error(f"[{date.date()}] ❌ Exception during trading day: {e}")
                logger.debug(traceback.format_exc())

            progress.advance(task)

    logger.info(f"✅ Backtest complete — {len(logs)} trading days processed")
    inspect_equity_spikes(logs)

    if trade_records:
        df_trades = pd.DataFrame(trade_records)
        trades_path = output_dir / "trades.csv"
        df_trades.sort_values(by=["date", "symbol"], inplace=True)
        df_trades.to_csv(trades_path, index=False)
        logger.info(f"📄 Saved trades to {trades_path}")

    return logs


def inspect_equity_spikes(logs: list[DailyLog], threshold: float = 30.0):
    previous_equity = None

    for log in logs:
        equity = float(log.equity or 0.0)
        date = log.date.date()
        pnl = equity - previous_equity if previous_equity is not None else 0.0

        if pnl > threshold:
            print(f"\n⚠️ Spike on {date} — PnL: ${pnl:.2f}, Equity: ${equity:.2f}")
            print(f"Cash: ${log.cash:.2f} | Trades: {log.trades.astype(bool).sum()}")

            if log.trades is not None and not log.trades.empty:
                top = log.trades.abs().sort_values(ascending=False).head(5)
                for symbol in top.index:
                    weight = log.trades[symbol]
                    print(f" - {symbol}: weight={weight:.4f}")
            else:
                print("No trades recorded.")

        previous_equity = equity
