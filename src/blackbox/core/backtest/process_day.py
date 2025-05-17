import pandas as pd

from blackbox.core.types.context import StrategyContext
from blackbox.core.types.types import (
    DailyLog,
    OHLCVSnapshot,
    PortfolioTarget,
    TradeResult,
)
from blackbox.utils.trade import reconcile_trades, simulate_execution


def process_trading_day(
    snapshot: OHLCVSnapshot,
    features_today: pd.DataFrame,
    features_window: pd.DataFrame,
    ctx: StrategyContext,
) -> DailyLog:
    """
    Process a single trading day:
    1. Execute pending orders from yesterday using today's open.
    2. Generate alpha signals for next day.
    3. Apply risk and cost adjustments.
    4. Create portfolio target and record orders.
    """
    try:
        date = snapshot.date
        logger = ctx.logger
        logger.debug(f"[{date.date()}] ➤ Starting trading day")

        # ════════════════════════════════════════════════
        # 1️⃣ EXECUTE PENDING ORDERS AT OPEN
        # ════════════════════════════════════════════════
        execution_result: TradeResult = None

        pending_orders = getattr(ctx.tracker, "get_pending_orders", lambda: None)()

        if isinstance(pending_orders, pd.DataFrame) and not pending_orders.empty:
            logger.debug(
                f"[{date.date()}] Executing {len(pending_orders)} pending orders"
            )

            decision_prices = (
                pending_orders["decision_price"]
                if "decision_price" in pending_orders
                else snapshot.open
            )

            capital_at_open = ctx.tracker.compute_portfolio_value(snapshot.open)

            execution_result = simulate_execution(
                trades=pending_orders["trade"],
                prices=snapshot.open,
                decision_prices=decision_prices,
                slippage=0.001,
                capital=capital_at_open,
                logger=logger,
                date=date,
            )

            ctx.tracker.update(execution_result.executed, execution_result.fill_prices)
            logger.debug(f"[{date.date()}] Portfolio updated after execution")

        # ════════════════════════════════════════════════
        # 2️⃣ GENERATE SIGNALS FOR NEXT DAY
        # ════════════════════════════════════════════════
        capital_at_close = ctx.tracker.compute_portfolio_value(snapshot.close)
        if ctx.config.verbose:
            logger.info(f"[{date.date()}] Current portfolio value: {capital_at_close}")

        raw_signals = ctx.models.alpha.predict(features_today)
        logger.debug(f"[{date.date()}] RAW_SIGNALS:\n{raw_signals[raw_signals != 0]}")

        risk_adjusted = ctx.models.risk.apply(raw_signals, features_window)
        logger.debug(
            f"[{date.date()}] RISK_ADJUSTED:\n{risk_adjusted[risk_adjusted != 0]}"
        )

        cost_adjusted = ctx.models.cost.adjust(risk_adjusted, features_window)
        logger.debug(
            f"[{date.date()}] COST_ADJUSTED:\n{cost_adjusted[cost_adjusted != 0]}"
        )

        target: PortfolioTarget = ctx.models.portfolio.construct(
            cost_adjusted, capital_at_close, features_window, snapshot
        )
        logger.debug(f"[{date.date()}] Portfolio target:\n{target}")

        # ════════════════════════════════════════════════
        # 3️⃣ RECONCILE TRADES FOR NEXT DAY
        # ════════════════════════════════════════════════
        current_portfolio = ctx.tracker.get_portfolio()
        required_trades = reconcile_trades(current_portfolio, target.weights)

        if hasattr(ctx.tracker, "record_pending_orders"):
            trades_with_metadata = pd.DataFrame(
                {
                    "trade": required_trades,
                    "decision_price": snapshot.close.reindex(required_trades.index),
                }
            )

            ctx.tracker.record_pending_orders(trades_with_metadata)
            logger.debug(
                f"[{date.date()}] Recorded {len(required_trades)} orders for tomorrow"
            )

        # ════════════════════════════════════════════════
        # 4️⃣ LOGGING & REPORTING
        # ════════════════════════════════════════════════
        equity = ctx.tracker.compute_portfolio_value(snapshot.close)
        cash = ctx.tracker.current_cash
        pnl = equity + cash - ctx.initial_equity
        drawdown = (equity + cash) / ctx.initial_equity - 1

        executed_trades = (
            pd.Series(0.0, index=snapshot.close.index)
            if execution_result is None
            else execution_result.executed
        )
        feedback = {} if execution_result is None else execution_result.feedback

        return DailyLog(
            date=date,
            prices=snapshot.close,
            trades=executed_trades,
            portfolio=ctx.tracker.get_portfolio(),
            feedback=feedback,
            equity=equity,
            cash=cash,
            pnl=pnl,
            drawdown=drawdown,
        )
    except Exception as e:
        logger.exception(f"[{snapshot.date}] ⚠️ Exception in trading day: {e}")
        return None
