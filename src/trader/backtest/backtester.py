import logging
import os
from collections import defaultdict
from typing import Tuple

import pandas as pd
from tqdm import tqdm

from trader.backtest.backtest_config import BacktestConfig
from trader.core.engine import ModelBundle
from trader.utils.alpha_diagnostics_logger import AlphaDiagnosticsLogger

logger = logging.getLogger(__name__)


class Backtester:
    def __init__(self, models: ModelBundle, config: BacktestConfig):
        self.models = models
        self.config = config
        self.cash = config.initial_capital
        self.positions = defaultdict(int)
        self.position_age = defaultdict(int)
        self.history = []
        self.daily_metrics = []
        self.diagnostics = None

    def run(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Starting backtest...")
        df = data.copy()
        df["symbol"] = df["symbol"].astype(str)

        output_dir = self.config.results_dir or "results/default"
        os.makedirs(output_dir, exist_ok=True)
        self.diagnostics = AlphaDiagnosticsLogger(output_dir)

        if "timestamp" not in df.columns:
            if "day" in df.columns:
                df = df.rename(columns={"day": "timestamp"})
                logger.info("⏱️ Renamed 'day' column to 'timestamp'")
            else:
                raise KeyError(
                    "Input DataFrame must contain a 'timestamp' or 'day' column."
                )

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        if "price" not in df.columns:
            if "close" in df.columns:
                df["price"] = df["close"]
                logger.info("📈 Set 'price' column from 'close'")
            else:
                raise ValueError("Missing both 'price' and 'close' columns.")

        df = df.sort_values(["symbol", "timestamp"]).dropna(subset=["price"])
        unique_days = df["timestamp"].dt.floor("D").drop_duplicates().sort_values()
        if self.config.warmup_period > 0:
            unique_days = unique_days[self.config.warmup_period :]

        max_equity_so_far = self.cash
        min_holding_days = 3

        for current_day in tqdm(unique_days, desc="Backtest", unit="day"):
            tqdm.write(f"📆 Processing {current_day.date()}")
            prev_positions = self.positions.copy()
            daily_df = df[df["timestamp"].dt.floor("D") == current_day]
            if daily_df.empty:
                continue

            for current_time in daily_df["timestamp"].drop_duplicates().sort_values():
                snapshot = daily_df[daily_df["timestamp"] == current_time][
                    ["symbol", "price", "volume"]
                ]
                window = df[df["timestamp"] < current_time]

                alpha_df = self.models.alpha.score(window, current_time)
                self.diagnostics.log_daily_alpha(alpha_df, current_time)

                risk_df = self.models.risk.score(window, current_time)

                current_prices = snapshot.set_index("symbol")["price"]
                portfolio_value = sum(
                    self.positions[sym] * current_prices.get(sym, 0.0)
                    for sym in self.positions
                )
                total_equity = self.cash + portfolio_value
                max_equity_so_far = max(max_equity_so_far, total_equity)
                drawdown = (
                    (total_equity / max_equity_so_far - 1.0)
                    if max_equity_so_far
                    else 0.0
                )

                self.daily_metrics.append(
                    {
                        "timestamp": current_time,
                        "cash": self.cash,
                        "portfolio_value": portfolio_value,
                        "equity": total_equity,
                        "drawdown": drawdown,
                    }
                )

                current_pos_df = pd.DataFrame(
                    {
                        "symbol": list(self.positions.keys()),
                        "shares": list(self.positions.values()),
                    }
                )
                current_pos_df["price"] = (
                    current_pos_df["symbol"].map(current_prices).fillna(0)
                )
                tx_df = self.models.tx_cost.score(current_pos_df, current_time)
                slippage_df = self.models.slippage.score(current_pos_df, current_time)

                eligible_symbols = alpha_df["symbol"].unique()
                eligible_symbols = [
                    sym
                    for sym in eligible_symbols
                    if self.position_age[sym] >= min_holding_days
                    or self.positions[sym] == 0
                ]
                alpha_df = alpha_df[alpha_df["symbol"].isin(eligible_symbols)]

                targets = self.models.portfolio.allocate(
                    alpha_df=alpha_df,
                    risk_df=risk_df,
                    tx_df=tx_df,
                    slippage_df=slippage_df,
                    price_df=snapshot,
                    capital=total_equity,
                )

                if targets.empty:
                    tqdm.write("⚠️  No targets generated.")
                    continue

                trades = []
                for _, row in targets.iterrows():
                    symbol = row["symbol"]
                    target_shares = row["shares"]
                    current_shares = self.positions.get(symbol, 0)
                    delta = target_shares - current_shares
                    if delta != 0:
                        trades.append(
                            {
                                "symbol": symbol,
                                "shares": abs(delta),
                                "side": "buy" if delta > 0 else "sell",
                            }
                        )

                if not trades:
                    tqdm.write("➖ No change in portfolio today.")
                    continue

                orders_df = pd.DataFrame(trades)
                fills = self.models.execution.execute(
                    orders_df, snapshot, current_time, self.cash
                )

                for fill in fills:
                    symbol = fill["symbol"]
                    qty = fill["quantity"]
                    price = fill["fill_price"]
                    slippage = fill["slippage"]
                    cost = qty * price + slippage

                    if fill["side"] == "buy":
                        self.positions[symbol] += qty
                        self.cash -= cost
                    else:
                        self.positions[symbol] -= qty
                        self.cash += qty * price - slippage

                    self.history.append(
                        {**fill, "cash": self.cash, "timestamp": current_time}
                    )

                position_deltas = []
                for sym in set(self.positions.keys()).union(prev_positions.keys()):
                    prev_qty = prev_positions.get(sym, 0)
                    curr_qty = self.positions.get(sym, 0)
                    delta = curr_qty - prev_qty
                    if delta != 0:
                        side = "buy" if delta > 0 else "sell"
                        position_deltas.append((sym, side, abs(delta)))

                if getattr(self.config, "log_positions", False):
                    for sym, side, qty in position_deltas:
                        tqdm.write(f"📦 Position change: {side} {qty} of {sym}")

            for sym in list(self.positions):
                if self.positions[sym] > 0:
                    self.position_age[sym] += 1
                else:
                    self.position_age[sym] = 0

            tqdm.write(
                f"💰 {current_day.date()} | Cash: {self.cash:,.2f}, "
                f"Equity: {total_equity:,.2f}, Drawdown: {drawdown:.2%}\n"
            )

        result_df = pd.DataFrame(self.history)
        equity_curve = (
            pd.DataFrame(self.daily_metrics)
            .set_index("timestamp")["equity"]
            .sort_index()
            .ffill()
        )

        self.diagnostics.save()
        self.diagnostics.analyze()

        return result_df, equity_curve
