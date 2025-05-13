from typing import Iterator, Protocol

import pandas as pd

from blackbox.core.execution_loop import reconcile_trades, simulate_execution
from blackbox.models.interfaces import (AlphaModel, ExecutionModel,
                                        PortfolioConstructionModel, RiskModel,
                                        TransactionCostModel)
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
    ):
        self.alpha = alpha
        self.risk = risk
        self.cost = cost
        self.portfolio = portfolio
        self.execution = execution
        self.logger = logger
        self.portfolio_state = pd.Series(dtype=float)
        self.history = []

    def run(self, stream: LiveDataStream):
        for snapshot in stream:
            date = snapshot["date"]
            prices = snapshot["prices"]

            signals = self.alpha.generate(snapshot)
            risk_adjusted = self.risk.apply(signals, self.portfolio_state)
            cost_adjusted = self.cost.adjust(risk_adjusted, self.portfolio_state)
            target = self.portfolio.construct(cost_adjusted)

            trades = reconcile_trades(self.portfolio_state, target)
            executed, feedback = simulate_execution(trades, prices)

            self.execution.record(executed, feedback)
            self.portfolio.feedback_from_execution(feedback)
            self.portfolio_state = self.execution.update_portfolio(
                self.portfolio_state, executed
            )

            self.logger.info(f"{date.date()} | {len(executed)} trades executed")

            self.history.append(
                {
                    "date": date,
                    "portfolio": self.portfolio_state.copy(),
                    "trades": executed.copy(),
                    "prices": prices.copy(),
                }
            )

    def generate_metrics(self) -> dict:
        df = pd.DataFrame(self.history).set_index("date")
        return PerformanceMetrics().compute(df)
