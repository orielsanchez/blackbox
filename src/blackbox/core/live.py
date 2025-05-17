from typing import Iterator, Protocol

import pandas as pd

from blackbox.core.backtest.process_day import reconcile_trades, simulate_execution
from blackbox.core.types.types import TradeResult
from blackbox.models.interfaces import (
    AlphaModel,
    ExecutionModel,
    PortfolioConstructionModel,
    RiskModel,
    TransactionCostModel,
)
from blackbox.research.metrics import PerformanceMetrics
from blackbox.utils.logger import RichLogger


class LiveSnapshot(Protocol):
    def __getitem__(self, key: str) -> object: ...


class LiveDataStream(Protocol):
    def __iter__(self) -> Iterator[LiveSnapshot]: ...


class LiveTradingEngine:
    def __init__(
        self,
        alpha: AlphaModel,
        risk: RiskModel,
        cost: TransactionCostModel,
        portfolio: PortfolioConstructionModel,
        execution: ExecutionModel,
        logger: RichLogger,
        *,
        slippage: float = 0.001,
        initial_equity: float = 1_000_000.0,
    ) -> None:
        self.alpha = alpha
        self.risk = risk
        self.cost = cost
        self.portfolio = portfolio
        self.execution = execution
        self.logger = logger
        self.slippage = slippage
        self.initial_equity = initial_equity

        self.portfolio_state = pd.Series(dtype=float)
        self.history: list[dict[str, object]] = []

        self.execution.portfolio_value = initial_equity
        self.execution.current_cash = initial_equity

    def run(self, stream: LiveDataStream) -> None:
        for snapshot in stream:
            date = snapshot["date"]
            prices = snapshot["prices"]

            try:
                # Generate signal and adjust
                signals = self.alpha.generate(snapshot)
                risk_adjusted = self.risk.apply(signals, self.portfolio_state)
                cost_adjusted = self.cost.adjust(risk_adjusted, self.portfolio_state)
                target = self.portfolio.construct(cost_adjusted)

                trades = reconcile_trades(self.portfolio_state, target)

                # Snapshot current state
                equity = getattr(self.execution, "portfolio_value", self.initial_equity)
                cash = getattr(self.execution, "current_cash", equity * 0.5)

                outcome: TradeOutcome = simulate_execution(
                    trades=trades,
                    prices=prices,
                    slippage=self.slippage,
                    capital=equity,
                    logger=self.logger,
                    initial_equity=self.initial_equity,
                    execution_state={"portfolio_value": equity, "current_cash": cash},
                    date=date,
                )

                # Log and apply result
                self.execution.record(outcome.result.executed, outcome.result.feedback)
                self.portfolio.feedback_from_execution(outcome.result.feedback)

                self.portfolio_state = self.execution.update_portfolio(
                    self.portfolio_state,
                    outcome.result.executed,
                    outcome.equity,
                    prices,
                )
                self.execution.mark_to_market(prices)

                self.logger.info(
                    f"{date.date()} | ✅ {len(outcome.result.executed)} trades executed"
                )

                self.history.append(
                    {
                        "date": date,
                        "portfolio": self.portfolio_state.copy(),
                        "trades": outcome.result.executed.copy(),
                        "prices": prices.copy(),
                        "equity": outcome.equity,
                        "cash": outcome.cash,
                    }
                )

            except Exception as e:
                self.logger.error(f"{date.date()} | ⚠️ Live error: {e}", exc_info=True)

    def generate_metrics(self) -> dict:
        if not self.history:
            self.logger.warning("⚠️ No live history to evaluate.")
            return {}

        df = pd.DataFrame(self.history).set_index("date")
        return PerformanceMetrics(initial_value=self.initial_equity).compute(df)
