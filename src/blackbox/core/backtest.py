import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from blackbox.config.loader import dump_config
from blackbox.config.schema import BacktestConfig
from blackbox.core.execution_loop import DailyLog, TradeResult, reconcile_trades, simulate_execution
from blackbox.models.interfaces import (
    AlphaModel,
    ExecutionModel,
    PortfolioConstructionModel,
    RiskModel,
    TransactionCostModel,
)
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
        initial_equity: float = None,
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
        self.initial_equity = initial_equity or config.initial_portfolio_value
        self.risk_free_rate = risk_free_rate
        self.plot_equity = plot_equity
        self.daily_logs: list[DailyLog] = []
        # Set the execution model's portfolio value to match
        if hasattr(execution, "portfolio_value"):
            execution.portfolio_value = self.initial_equity
        if hasattr(execution, "current_cash"):
            execution.current_cash = self.initial_equity

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.output_dir = Path("results") / config.run_id / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dump_config(config, self.output_dir / "config.yaml")

    def run(self, data: list[dict], feature_matrix: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        # Track account equity by date
        self.equity_by_date = {}

        self.logger.info(f"📊 Backtest start | {len(data)} trading days")

        if feature_matrix is None:
            self.logger.warning("⚠️ No feature matrix passed — using global context")
            feature_matrix = get_feature_matrix()

        # Ensure proper MultiIndex
        if feature_matrix.index.names != ["date", "symbol"]:
            feature_matrix = (
                feature_matrix.reset_index()
                .assign(date=lambda df: pd.to_datetime(df["date"]).dt.normalize())
                .set_index(["date", "symbol"])
                .sort_index()
            )

        # Force MultiIndex to be unique, sorted, and correct type
        feature_matrix.index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(date), symbol) for date, symbol in feature_matrix.index],
            names=["date", "symbol"],
        )
        feature_matrix = feature_matrix.sort_index()
        assert feature_matrix.index.is_unique, "Feature matrix index is not unique!"

        # Safely sort unique dates
        feature_dates = (
            pd.to_datetime(feature_matrix.index.get_level_values("date"))
            .normalize()
            .unique()
            .sort_values()
        )

        self.logger.info(
            f"🕵️ Feature matrix covers {len(feature_dates)} dates | Sample: {feature_dates[:5]}"
        )
        self.logger.info(f"Feature matrix index sample: {feature_matrix.index[:5]}")
        self.logger.info(
            f"Feature matrix unique dates: {feature_matrix.index.get_level_values('date').unique()[:5]}"
        )

        # Defensive: Compare data dates and feature matrix dates
        feature_matrix_dates = set(feature_matrix.index.get_level_values("date").unique())
        data_dates = set(pd.to_datetime([snap["date"] for snap in data]).normalize().unique())
        missing_in_features = sorted(data_dates - feature_matrix_dates)
        if missing_in_features:
            self.logger.warning(
                f"⚠️ {len(missing_in_features)} dates in data but missing in feature matrix: {missing_in_features[:10]} ..."
            )

        # Determine warmup period based on when feature data becomes available
        min_feature_date = min(feature_matrix_dates)
        data_with_dates = [(pd.to_datetime(snap["date"]).normalize(), snap) for snap in data]
        data_with_dates.sort(key=lambda x: x[0])  # Sort by date

        # Determine the first date when we have feature data
        warmup_days = sum(1 for date, _ in data_with_dates if date < min_feature_date)

        if warmup_days > 0:
            self.logger.info(
                f"⏳ First {warmup_days} days are warmup (before {min_feature_date.date()})"
            )

        # Construct full OHLCV dataframe for easier lookups
        # FIX: Handle the case where date column already exists in the index or DataFrame
        frames = [
            snap["ohlcv"]
            for snap in data
            if "ohlcv" in snap
            and isinstance(snap["ohlcv"], pd.DataFrame)
            and not snap["ohlcv"].empty
        ]

        if frames:
            # Check the structure of the frames to determine how to process them
            sample_frame = frames[0]
            self.logger.debug(f"Sample OHLCV frame columns: {sample_frame.columns.tolist()}")
            self.logger.debug(
                f"Sample OHLCV frame index: {type(sample_frame.index)} - {sample_frame.index.names}"
            )

            # Correctly handle different possible structures
            if isinstance(sample_frame.index, pd.MultiIndex) and sample_frame.index.names == [
                "date",
                "symbol",
            ]:
                # Already has the correct MultiIndex
                self.logger.debug("OHLCV already has correct MultiIndex")
                ohlcv_df = pd.concat(frames)
            elif "date" in sample_frame.columns and "symbol" in sample_frame.columns:
                # Flat DataFrame with date and symbol columns
                self.logger.debug("OHLCV has date and symbol as columns")
                ohlcv_df = pd.concat(frames).set_index(["date", "symbol"])
            elif isinstance(sample_frame.index, pd.MultiIndex):
                # Has a MultiIndex but different names
                self.logger.debug(f"OHLCV has MultiIndex with names: {sample_frame.index.names}")
                # Rename index levels if needed and combine
                frames_with_correct_names = []
                for frame in frames:
                    if frame.index.names != ["date", "symbol"]:
                        frame = frame.copy()
                        frame.index.names = ["date", "symbol"]
                    frames_with_correct_names.append(frame)
                ohlcv_df = pd.concat(frames_with_correct_names)
            else:
                # Some other structure - try adding date from snapshot
                self.logger.debug("OHLCV has non-standard structure, adding date from snapshots")
                new_frames = []
                for i, frame in enumerate(frames):
                    date = pd.to_datetime(data[i]["date"]).normalize()
                    if not isinstance(frame.index, pd.MultiIndex):
                        # Assume index is symbol
                        frame = frame.copy()
                        frame.index.name = "symbol"
                        frame = frame.reset_index()
                        frame["date"] = date
                        frame = frame.set_index(["date", "symbol"])
                    new_frames.append(frame)
                ohlcv_df = pd.concat(new_frames)

            # Ensure all dates are normalized
            if isinstance(ohlcv_df.index, pd.MultiIndex):
                dates = ohlcv_df.index.get_level_values("date")
                symbols = ohlcv_df.index.get_level_values("symbol")
                ohlcv_df.index = pd.MultiIndex.from_tuples(
                    [(pd.to_datetime(d).normalize(), s) for d, s in zip(dates, symbols)],
                    names=["date", "symbol"],
                )
                ohlcv_df = ohlcv_df.sort_index()

            self.logger.info(f"Combined OHLCV dataframe: {ohlcv_df.shape} rows")
        else:
            ohlcv_df = pd.DataFrame()
            self.logger.warning("⚠️ No OHLCV data found in snapshots")

        # Enhanced validation - check if each snapshot has required data
        for i, snapshot in enumerate(data):
            date = pd.to_datetime(snapshot["date"]).normalize()

            # Debug: log what's in the snapshot
            if i < 5:  # Just log first few days to avoid spamming
                keys = list(snapshot.keys())
                self.logger.debug(f"Snapshot {date.date()} contains keys: {keys}")

                # Check if "prices" is present and has symbols
                if "prices" in snapshot:
                    num_prices = len(snapshot["prices"])
                    self.logger.debug(f"Snapshot {date.date()} has {num_prices} price symbols")
                else:
                    self.logger.warning(f"Snapshot {date.date()} missing 'prices'")

                # Check if "ohlcv" is present and properly structured
                if "ohlcv" in snapshot:
                    if isinstance(snapshot["ohlcv"], pd.DataFrame):
                        num_ohlcv = len(snapshot["ohlcv"])
                        self.logger.debug(
                            f"Snapshot {date.date()} has {num_ohlcv} OHLCV rows, columns: {snapshot['ohlcv'].columns.tolist()}"
                        )
                    else:
                        self.logger.warning(
                            f"Snapshot {date.date()} has invalid 'ohlcv' type: {type(snapshot['ohlcv'])}"
                        )
                else:
                    self.logger.warning(f"Snapshot {date.date()} missing 'ohlcv' key")

            # Fix missing OHLCV if we have the combined dataframe
            if (
                "ohlcv" not in snapshot
                or snapshot["ohlcv"] is None
                or (isinstance(snapshot["ohlcv"], pd.DataFrame) and snapshot["ohlcv"].empty)
            ):
                if not ohlcv_df.empty:
                    try:
                        # Extract for this date
                        date_data = ohlcv_df.loc[date]
                        snapshot["ohlcv"] = date_data
                        self.logger.debug(f"Added OHLCV data to snapshot for {date.date()}")
                    except KeyError:
                        self.logger.warning(f"No OHLCV data available for {date.date()}")

        equity_so_far = self.initial_equity
        any_trades_executed = False
        positions_built = False

        with Progress(
            TextColumn("[bold green]📅 {task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Backtesting", total=len(data))

            for snapshot in data:
                date = pd.to_datetime(snapshot["date"]).normalize()
                prices = snapshot["prices"]

                # Progress bar description shows current date and equity
                if equity_so_far is not None:
                    progress.update(task, description=f"{date.date()} | ${equity_so_far:.2f}")
                else:
                    progress.update(task, description=f"{date.date()}")

                progress.advance(task)

                try:
                    self.logger.debug(
                        f"feature_matrix.index.is_unique: {feature_matrix.index.is_unique}"
                    )
                    self.logger.debug(
                        f"feature_matrix.index.is_monotonic_increasing: {feature_matrix.index.is_monotonic_increasing}"
                    )

                    # Check if in warmup period
                    is_warmup = date < min_feature_date
                    if is_warmup:
                        self.logger.info(f"{date.date()} | 🔄 Warmup day (no features yet)")
                        continue

                    # Check if date exists in feature matrix
                    if date not in feature_matrix_dates:
                        self.logger.warning(
                            f"{date.date()} | ⚠️ Date missing from feature matrix, skipping."
                        )
                        continue

                    # Get features for this date
                    try:
                        snapshot["feature_vector"] = feature_matrix.loc[date]
                    except KeyError as e:
                        self.logger.error(f"KeyError for date {date}: {e}")
                        raise

                    # Calculate alpha signals
                    signals = self.alpha.predict(snapshot)
                    if signals.empty:
                        self.logger.warning(f"{date.date()} | ⚠️ No alpha signals generated")
                        continue

                    top_signals = signals.sort_values(ascending=False).head(5).to_dict()
                    self.logger.info(
                        f"{date.date()} | Alpha: {len(signals)} signals | Top: {top_signals}"
                    )

                    # Build portfolio
                    # Add capital to the snapshot
                    snapshot["capital"] = equity_so_far
                    positions = self.portfolio.construct(signals, snapshot)

                    # Track if we've built positions (out of warmup)
                    if not positions_built and not positions.empty:
                        positions_built = True

                    # Log exposure
                    gross_exposure = abs(positions).sum() if isinstance(positions, pd.Series) else 0
                    position_count = len(positions) if isinstance(positions, pd.Series) else 0
                    self.logger.info(
                        f"✅ Constructed {position_count} positions | Gross exposure: {gross_exposure:.2f}"
                    )

                    # Execute trades
                    snapshot["capital"] = equity_so_far  # Ensure capital is in snapshot
                    trades = self._simulate_day(date, snapshot, positions)

                    # Track whether any trades have been executed
                    if trades is not None and not any_trades_executed and not trades.empty:
                        any_trades_executed = True

                    # Update equity from internal tracking
                    if date in self.equity_by_date:
                        equity_so_far = self.equity_by_date[date]

                    if equity_so_far <= 0:
                        self.logger.error(f"Capital is zero or negative: ${equity_so_far:.2f}")
                        raise ValueError("Capital is zero")

                except Exception as e:
                    self.logger.error(f"{date.date()} | ⚠️ Exception: {e}")
                    self.logger.error(traceback.format_exc())

        if not self.daily_logs:
            self.logger.error("❌ No trades executed — likely due to data issues")
            return pd.DataFrame()

        self.logger.info(f"✅ Backtest completed | {len(self.daily_logs)} days recorded")

        if self.plot_equity:
            from blackbox.utils.plotting import plot_equity_curve

            plot_equity_curve(self.daily_logs, self.config.run_id, self.output_dir)

        return pd.DataFrame([log.__dict__ for log in self.daily_logs]).set_index("date")

    def _simulate_day(self, date: pd.Timestamp, snapshot: dict, positions: pd.Series):
        # Get prices from the snapshot
        prices = snapshot["prices"]

        # Re-generate signals to ensure consistency
        signals = self.alpha.generate(snapshot)
        tradable = signals.index.intersection(prices.index)

        if tradable.empty:
            self.logger.warning(f"{date.date()} | No tradable signals")
            return

        signals = signals.loc[tradable]
        nonzero = signals[signals != 0]
        top = nonzero.abs().sort_values(ascending=False).head(5)

        self.logger.info(f"{date.date()} | Alpha: {len(nonzero)} signals | Top: {top.to_dict()}")

        current_portfolio = self.tracker.get_portfolio()
        risk_adjusted = self.risk.apply(signals, current_portfolio)
        self._log_state("Risk-adjusted", date, risk_adjusted)

        # DEBUG: Added detailed risk adjustment info
        self.logger.info(
            f"{date.date()} | From Alpha to Risk: {signals.abs().sum():.4f} → {risk_adjusted.abs().sum():.4f}"
        )

        cost_adjusted = self.cost.adjust(risk_adjusted, current_portfolio)
        self._log_state("Cost-adjusted", date, cost_adjusted)

        # DEBUG: Added detailed cost adjustment info
        self.logger.info(
            f"{date.date()} | From Risk to Cost: {risk_adjusted.abs().sum():.4f} → {cost_adjusted.abs().sum():.4f}"
        )

        # Make sure snapshot has capital value for portfolio construction
        if "capital" not in snapshot or snapshot["capital"] is None:
            snapshot["capital"] = self.tracker.get_portfolio_value() or self.initial_equity
            self.logger.debug(f"Added missing capital to snapshot: ${snapshot['capital']:.2f}")

        target_portfolio = self.portfolio.construct(cost_adjusted, snapshot)
        self._log_state("Target", date, target_portfolio)

        # DEBUG: Added detailed portfolio construction info
        self.logger.info(
            f"{date.date()} | From Cost to Target: {cost_adjusted.abs().sum():.4f} → {target_portfolio.abs().sum():.4f}"
        )

        trades = reconcile_trades(current_portfolio, target_portfolio)
        self._log_state("Reconciled", date, trades)

        # DEBUG: Added detailed reconciliation info
        self.logger.info(
            f"{date.date()} | Reconciled: {len(trades)} trades | Notional: {trades.abs().sum():.4f}"
        )

        if trades.empty:
            self.logger.warning(
                f"{date.date()} | No trades to execute. Current portfolio size: {len(current_portfolio)}, Target size: {len(target_portfolio)}"
            )
            return

        trade_result: TradeResult = simulate_execution(
            trades,
            prices,
            slippage=self.slippage,
            capital=self.execution.portfolio_value,
        )

        self._log_state("Executed", date, trade_result.executed)
        self.logger.debug(f"{date.date()} | Feedback: {trade_result.feedback}")

        filtered = self.tracker.filter(trade_result.executed, date, self.min_holding)
        self._log_state("Filtered", date, filtered)

        self.execution.record(filtered, trade_result.feedback)
        updated = self.execution.update_portfolio(current_portfolio, filtered)

        self.tracker.update(updated, date)
        self.portfolio.feedback_from_execution(trade_result.feedback)

        self.logger.info(f"{date.date()} | {len(filtered)} trades executed")

        # Create daily log entry
        daily_log = DailyLog(
            date=date,
            prices=prices.copy(),
            trades=filtered.copy(),
            portfolio=updated.copy(),
            feedback=trade_result.feedback.copy(),
        )

        # Track equity for this date
        # This is a simplified implementation - in reality would calculate actual P&L
        self.equity_by_date[date] = self.initial_equity

        self.daily_logs.append(daily_log)

        return filtered

    def _log_state(self, label: str, date: pd.Timestamp, series: pd.Series):
        nonzero = series[series != 0]
        if not nonzero.empty:
            self.logger.debug(f"{date.date()} | {label}: {nonzero.to_dict()}")

    def generate_metrics(self, return_equity: bool = False) -> dict:
        if not self.daily_logs:
            self.logger.error("❌ No daily logs available for metrics.")
            return {}

        df = pd.DataFrame([log.__dict__ for log in self.daily_logs]).set_index("date")
        metrics = PerformanceMetrics(
            initial_value=self.initial_equity,
            risk_free_rate=self.risk_free_rate,
        )
        return metrics.compute(df, return_equity=return_equity)
