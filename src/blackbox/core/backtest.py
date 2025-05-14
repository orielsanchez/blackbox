from typing import Optional

import pandas as pd

from blackbox.core.execution_loop import (DailyLog, TradeResult,
                                          reconcile_trades, simulate_execution)
from blackbox.models.interfaces import (AlphaModel, ExecutionModel,
                                        PortfolioConstructionModel, RiskModel,
                                        TransactionCostModel)
from blackbox.models.tracker import PositionTracker
from blackbox.research.metrics import PerformanceMetrics
from blackbox.utils.context import get_feature_matrix, get_logger
from blackbox.utils.logger import RichLogger


class BacktestEngine:
    def __init__(
        self,
        alpha: AlphaModel,
        risk: RiskModel,
        cost: TransactionCostModel,
        portfolio: PortfolioConstructionModel,
        execution: ExecutionModel,
        logger: Optional[RichLogger] = None,
        position_tracker: Optional[PositionTracker] = None,
        min_holding_period: int = 0,
        slippage: float = 0.001,
    ):
        self.alpha = alpha
        self.risk = risk
        self.cost = cost
        self.portfolio = portfolio
        self.execution = execution
        self.logger = logger or get_logger()

        self.tracker = position_tracker or PositionTracker()
        self.min_holding = min_holding_period
        self.slippage = slippage
        self.history: list[DailyLog] = []

    def run(self, data: list[dict]) -> pd.DataFrame:
        feature_matrix = get_feature_matrix()

        for snapshot in data:
            date: pd.Timestamp = snapshot["date"]
            prices: pd.Series = snapshot["prices"]

            snapshot["feature_matrix"] = feature_matrix

            # Simulate price movement and update capital
            self.execution.mark_to_market(prices)
            snapshot["capital"] = self.execution.portfolio_value

            # === Alpha
            signals = self.alpha.generate(snapshot)
            nonzero_signals = signals[signals != 0]
            top_signals = nonzero_signals.sort_values(key=abs, ascending=False).head(5)
            self.logger.info(
                f"{date.date()} alpha signals: {len(nonzero_signals)} non-zero | Top: {top_signals.to_dict()}"
            )

            # === Risk
            current_portfolio = self.tracker.get_portfolio()
            risk_adjusted = self.risk.apply(signals, current_portfolio)
            self.logger.debug(
                f"{date.date()} risk-adjusted: {risk_adjusted[risk_adjusted != 0].to_dict()}"
            )

            # === Cost
            cost_adjusted = self.cost.adjust(risk_adjusted, current_portfolio)
            self.logger.debug(
                f"{date.date()} cost-adjusted: {cost_adjusted[cost_adjusted != 0].to_dict()}"
            )

            # === Portfolio Construction
            target_portfolio = self.portfolio.construct(cost_adjusted, snapshot)
            self.logger.debug(
                f"{date.date()} target portfolio: {target_portfolio[target_portfolio != 0].to_dict()}"
            )

            # === Trades
            trades = reconcile_trades(current_portfolio, target_portfolio)
            self.logger.debug(
                f"{date.date()} raw trades: {trades[trades != 0].to_dict()}"
            )

            # === Simulate Execution
            trade_result: TradeResult = simulate_execution(
                trades,
                prices,
                slippage=self.slippage,
                capital=self.execution.portfolio_value,
            )

            self.logger.debug(
                f"{date.date()} executed trades: {trade_result.executed[trade_result.executed != 0].to_dict()}"
            )
            self.logger.debug(
                f"{date.date()} execution feedback: {trade_result.feedback}"
            )

            # === Filter based on holding period
            filtered_trades = self.tracker.filter(
                trade_result.executed, date, self.min_holding
            )
            self.logger.debug(
                f"{date.date()} filtered trades: {filtered_trades[filtered_trades != 0].to_dict()}"
            )

            # === Update
            self.execution.record(filtered_trades, trade_result.feedback)
            updated_portfolio = self.execution.update_portfolio(
                current_portfolio, filtered_trades
            )
            self.tracker.update(updated_portfolio, date)
            self.portfolio.feedback_from_execution(trade_result.feedback)

            self.logger.info(f"{date.date()} | {len(filtered_trades)} trades executed")

            self.history.append(
                DailyLog(
                    date=date,
                    prices=prices.copy(),
                    trades=filtered_trades.copy(),
                    portfolio=updated_portfolio.copy(),
                    feedback=trade_result.feedback.copy(),
                )
            )

        return pd.DataFrame([log.__dict__ for log in self.history]).set_index("date")

    def generate_metrics(self) -> dict:
        df = pd.DataFrame([log.__dict__ for log in self.history]).set_index("date")
        metrics = PerformanceMetrics()
        return metrics.compute(df)
