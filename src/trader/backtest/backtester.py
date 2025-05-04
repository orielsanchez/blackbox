import logging

import pandas as pd

from trader.backtest.backtest_config import BacktestConfig
from trader.core.engine import ModelBundle

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self, models: ModelBundle, config: BacktestConfig):
        self.models = models
        self.config = config
        self.cash = config.initial_capital
        self.positions = {}  # symbol -> quantity
        self.history = []
        self.execution_stats = {}

    def run(self, data: pd.DataFrame) -> pd.Series:
        logger.info("Starting unified backtest...")
        df = data.copy()

        if self.config.start_date:
            df = df[df["timestamp"] >= self.config.start_date]
        if self.config.end_date:
            df = df[df["timestamp"] <= self.config.end_date]

        df = df.sort_values("timestamp")
        unique_dates = df["timestamp"].unique()

        portfolio_value_series = []
        all_fills = []

        for idx, current_date in enumerate(unique_dates):
            if idx < self.config.warmup_period:
                continue

            logger.debug(f"Processing {current_date}")
            day_data = df[df["timestamp"] == current_date]

            alpha_scores = self.models.alpha.score(day_data)
            risk_scores = self.models.risk.score(day_data)
            cost_scores = self.models.tx_cost.score(day_data)

            targets = self.models.portfolio.allocate(
                alpha_scores=alpha_scores,
                risk_scores=risk_scores,
                cost_scores=cost_scores,
                capital=self.cash,
            )

            fills = self.models.execution.execute_orders(
                target_positions=targets, market_data=day_data
            )

            self._update_positions(fills)
            all_fills.append(fills)

            value = self._portfolio_value(day_data)
            portfolio_value_series.append((current_date, value))

        result_df = pd.DataFrame(portfolio_value_series, columns=["timestamp", "value"])
        equity_curve = result_df.set_index("timestamp")["value"]

        if all_fills:
            combined_fills = pd.concat(all_fills)
            self.execution_stats = {
                "avg_slippage_bps": combined_fills["slippage_pct"].mean(),
                "total_slippage": combined_fills["slippage"].sum(),
                "num_fills": len(combined_fills),
            }

        return equity_curve

    def _update_positions(self, fills: pd.DataFrame):
        for symbol, row in fills.iterrows():
            qty = row.get("quantity", 0)
            price = row.get("fill_price", 0.0)
            cost = qty * price
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
            self.cash -= cost

    def _portfolio_value(self, day_data: pd.DataFrame) -> float:
        value = self.cash
        price_map = day_data.set_index("symbol")["close"].to_dict()
        for symbol, qty in self.positions.items():
            price = price_map.get(symbol)
            if price is not None:
                value += qty * price
        return value
