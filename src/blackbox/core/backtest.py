from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from rich.progress import (BarColumn, Progress, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)

from blackbox.config.loader import dump_config
from blackbox.config.schema import BacktestConfig
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
        config: BacktestConfig,
        alpha: AlphaModel,
        risk: RiskModel,
        cost: TransactionCostModel,
        portfolio: PortfolioConstructionModel,
        execution: ExecutionModel,
        logger: Optional[RichLogger] = None,
        position_tracker: Optional[PositionTracker] = None,
        min_holding_period: int = 0,
        slippage: float = 0.001,
        initial_equity: float = 1_000_000,
        risk_free_rate: float = 0.0,
        plot_equity: bool = True,
    ):
        self.config = config
        self.alpha = alpha
        self.risk = risk
        self.cost = cost
        self.portfolio = portfolio
        self.execution = execution
        self.logger = logger or get_logger()
        self.tracker = position_tracker or PositionTracker()
        self.min_holding = min_holding_period
        self.slippage = slippage
        self.daily_logs: list[DailyLog] = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.output_dir = Path("results") / config.run_id / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_equity = initial_equity
        self.risk_free_rate = risk_free_rate
        self.plot_equity = plot_equity
        dump_config(self.config, self.output_dir / "config.yaml")

    def run(self, data: list[dict]) -> pd.DataFrame:
        full_feature_matrix = get_feature_matrix()

        with Progress(
            TextColumn("[bold green]\U0001f4c5 {task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Backtesting", total=len(data))

            for snapshot in data:
                date: pd.Timestamp = snapshot["date"]
                prices: pd.Series = snapshot["prices"]

                if not self._has_features_for_date(full_feature_matrix, date):
                    self.logger.warning(
                        f"{date.date()} | No features available — skipping"
                    )
                    progress.advance(task)
                    continue

                try:
                    snapshot["feature_vector"] = full_feature_matrix.loc[date]
                except KeyError:
                    self.logger.warning(
                        f"{date.date()} | Feature lookup failed — skipping"
                    )
                    progress.advance(task)
                    continue

                if prices.empty:
                    self.logger.warning(f"{date.date()} | No price data — skipping")
                    progress.advance(task)
                    continue

                self.execution.mark_to_market(prices)
                snapshot["capital"] = self.execution.portfolio_value

                try:
                    self._simulate_day(date, snapshot, prices)
                except Exception as e:
                    self.logger.error(f"❌ {date.date()} | Simulation failed: {e}")

                capital = snapshot.get("capital", 0)
                progress.update(
                    task, advance=1, description=f"{date.date()} | ${capital:,.2f}"
                )

        if not self.daily_logs:
            self.logger.error(
                "❌ No backtest logs recorded — likely due to missing data or capital."
            )
            return pd.DataFrame()

        if self.plot_equity:
            from blackbox.utils.plotting import plot_equity_curve

            plot_equity_curve(
                self.daily_logs, run_id=self.config.run_id, output_dir=self.output_dir
            )

        return pd.DataFrame([log.__dict__ for log in self.daily_logs]).set_index("date")

    def _has_features_for_date(
        self, features: pd.DataFrame, date: pd.Timestamp
    ) -> bool:
        return date in features.index.get_level_values("date")

    def _simulate_day(self, date: pd.Timestamp, snapshot: dict, prices: pd.Series):
        signals = self.alpha.generate(snapshot)

        tradable = signals.index.intersection(prices.index)
        if tradable.empty:
            self.logger.warning(f"{date.date()} | No tradable signals — skipping")
            return

        signals = signals.loc[tradable]
        nonzero_signals = signals[signals != 0]
        top_signals = nonzero_signals.abs().sort_values(ascending=False).head(5)

        self.logger.info(
            f"{date.date()} | Alpha signals: {len(nonzero_signals)} non-zero | Top: {top_signals.to_dict()}"
        )

        current_portfolio = self.tracker.get_portfolio()
        risk_adjusted = self.risk.apply(signals, current_portfolio)
        self._log_portfolio_state("Risk-adjusted", date, risk_adjusted)

        cost_adjusted = self.cost.adjust(risk_adjusted, current_portfolio)
        self._log_portfolio_state("Cost-adjusted", date, cost_adjusted)

        target_portfolio = self.portfolio.construct(cost_adjusted, snapshot)
        self._log_portfolio_state("Target portfolio", date, target_portfolio)

        trades = reconcile_trades(current_portfolio, target_portfolio)
        self._log_portfolio_state("Raw trades", date, trades)

        trade_result: TradeResult = simulate_execution(
            trades,
            prices,
            slippage=self.slippage,
            capital=self.execution.portfolio_value,
        )

        self._log_portfolio_state("Executed trades", date, trade_result.executed)
        self.logger.debug(
            f"{date.date()} | Execution feedback: {trade_result.feedback}"
        )

        filtered_trades = self.tracker.filter(
            trade_result.executed, date, self.min_holding
        )
        self._log_portfolio_state("Filtered trades", date, filtered_trades)

        self.execution.record(filtered_trades, trade_result.feedback)
        updated_portfolio = self.execution.update_portfolio(
            current_portfolio, filtered_trades
        )

        self.tracker.update(updated_portfolio, date)
        self.portfolio.feedback_from_execution(trade_result.feedback)

        self.logger.info(f"{date.date()} | {len(filtered_trades)} trades executed")

        self.daily_logs.append(
            DailyLog(
                date=date,
                prices=prices.copy(),
                trades=filtered_trades.copy(),
                portfolio=updated_portfolio.copy(),
                feedback=trade_result.feedback.copy(),
            )
        )

    def _log_portfolio_state(self, label: str, date: pd.Timestamp, series: pd.Series):
        nonzero = series[series != 0]
        if not nonzero.empty:
            self.logger.debug(f"{date.date()} | {label}: {nonzero.to_dict()}")

    def generate_metrics(self, return_equity: bool = False) -> dict:
        if not self.daily_logs:
            self.logger.error("❌ Cannot compute metrics — no daily logs available.")
            return {}

        df = pd.DataFrame([log.__dict__ for log in self.daily_logs]).set_index("date")

        metrics = PerformanceMetrics(
            initial_value=self.initial_equity,
            risk_free_rate=self.risk_free_rate,
        )
        return metrics.compute(df, return_equity=return_equity)
