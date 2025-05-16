import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """Daily‑step back‑test engine orchestrating Alpha → Risk → Cost → Portfolio → Execution."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
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
        *,
        min_holding_period: int = 0,
        slippage: float = 0.001,
        initial_equity: Optional[float] = None,
        risk_free_rate: float = 0.0,
        plot_equity: bool = True,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.alpha = alpha
        self.risk = risk
        self.cost = cost
        self.portfolio = portfolio
        self.execution = execution
        self.logger = logger or get_logger()
        self.tracker = position_tracker or PositionTracker()

        # Hyper‑parameters
        self.min_holding = min_holding_period
        self.slippage = slippage
        self.initial_equity = initial_equity or config.initial_portfolio_value
        self.risk_free_rate = risk_free_rate
        self.plot_equity = plot_equity
        self.verbose = verbose

        # Runtime series
        self.daily_logs: List[DailyLog] = []
        self.equity_by_date: dict[pd.Timestamp, float] = {}
        self.cash_by_date: dict[pd.Timestamp, float] = {}
        self.drawdown_by_date: dict[pd.Timestamp, float] = {}
        self.daily_pnl: dict[pd.Timestamp, float] = {}
        self.max_equity = self.initial_equity

        self._setup_execution_models()
        self._setup_output_directory()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _setup_execution_models(self) -> None:
        if hasattr(self.execution, "portfolio_value"):
            self.execution.portfolio_value = self.initial_equity
        if hasattr(self.execution, "current_cash"):
            self.execution.current_cash = self.initial_equity
            self.cash_by_date[pd.Timestamp.now().normalize()] = self.initial_equity

    def _setup_output_directory(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        self.output_dir = Path("results") / self.config.run_id / ts
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dump_config(self.config, self.output_dir / "config.yaml")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, data: List[Dict], feature_matrix: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.logger.info(f"📊 Backtest start | {len(data)} trading days")

        fm = self._prepare_feature_matrix(feature_matrix)
        fm_dates, min_fm_date, _ = self._analyze_features(data, fm)
        ohlcv_df = self._process_ohlcv_data(data)
        self._fix_missing_ohlcv(data, ohlcv_df)

        history_df = self._execute_backtest_days(data, fm, fm_dates, min_fm_date)

        if not self.daily_logs:
            self.logger.error("❌ No trades executed — likely data issues")
            return pd.DataFrame()

        self.logger.info(f"✅ Backtest completed | {len(self.daily_logs)} days recorded")
        if self.plot_equity:
            self._plot_equity_curve()
        return history_df

    def generate_metrics(self, *, return_equity: bool = False) -> dict:
        if not self.daily_logs:
            self.logger.error("❌ No daily logs available for metrics.")
            return {}
        df = pd.DataFrame([dl.__dict__ for dl in self.daily_logs]).set_index("date")
        return PerformanceMetrics(self.initial_equity, self.risk_free_rate).compute(
            df, return_equity=return_equity
        )

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _prepare_feature_matrix(self, fm: Optional[pd.DataFrame]) -> pd.DataFrame:
        if fm is None:
            self.logger.warning("⚠️ No feature matrix passed — using global context")
            fm = get_feature_matrix()
        if fm.index.names != ["date", "symbol"]:
            fm = (
                fm.reset_index()
                .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
                .set_index(["date", "symbol"])
                .sort_index()
            )
        fm.index = pd.MultiIndex.from_tuples(
            [(pd.to_datetime(d).normalize(), s) for d, s in fm.index.to_list()],
            names=["date", "symbol"],
        )
        assert fm.index.is_unique, "Feature matrix index not unique"
        return fm.sort_index()

    def _analyze_features(
        self, data: List[Dict], fm: pd.DataFrame
    ) -> Tuple[set[pd.Timestamp], pd.Timestamp, int]:
        fm_dates = set(pd.to_datetime(fm.index.get_level_values("date")).normalize().unique())
        min_fm_date = min(fm_dates)
        data_dates = set(pd.to_datetime([d["date"] for d in data]).normalize())
        warmup = sum(dt < min_fm_date for dt in data_dates)
        miss = sorted(data_dates - fm_dates)
        if miss and self.verbose:
            self.logger.warning(
                f"⚠️ {len(miss)} dates in data but missing in feature matrix: {miss[:10]} …"
            )
        return fm_dates, min_fm_date, warmup

    # ------------------------------------------------------------------
    # OHLCV helpers
    # ------------------------------------------------------------------
    def _process_ohlcv_data(self, data: List[Dict]) -> pd.DataFrame:
        frames = [
            d["ohlcv"]
            for d in data
            if isinstance(d.get("ohlcv"), pd.DataFrame) and not d["ohlcv"].empty
        ]
        if not frames:
            self.logger.warning("⚠️ No OHLCV data found in snapshots")
            return pd.DataFrame()
        return self._combine_ohlcv_frames(frames, data)

    def _combine_ohlcv_frames(self, frames: List[pd.DataFrame], data: List[Dict]) -> pd.DataFrame:
        first = frames[0]
        if isinstance(first.index, pd.MultiIndex) and first.index.names == [
            "date",
            "symbol",
        ]:
            ohlcv = pd.concat(frames)
        elif {"date", "symbol"}.issubset(first.columns):
            ohlcv = pd.concat(frames).set_index(["date", "symbol"])
        elif isinstance(first.index, pd.MultiIndex):
            fixed = [
                (
                    f.copy().set_axis(["date", "symbol"], axis=0, inplace=False)
                    if f.index.names != ["date", "symbol"]
                    else f
                )
                for f in frames
            ]
            ohlcv = pd.concat(fixed)
        else:
            built = []
            for snap, frame in zip(data, frames):
                date = pd.to_datetime(snap["date"]).normalize()
                g = frame.copy()
                g.index.name = "symbol"
                g = g.reset_index()
                g["date"] = date
                built.append(g.set_index(["date", "symbol"]))
            ohlcv = pd.concat(built)
        dates = ohlcv.index.get_level_values("date")
        syms = ohlcv.index.get_level_values("symbol")
        ohlcv.index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(dates).normalize(), syms], names=["date", "symbol"]
        )
        return ohlcv.sort_index()

    def _fix_missing_ohlcv(self, data: List[Dict], ohlcv: pd.DataFrame) -> None:
        if ohlcv.empty:
            return
        for snap in data:
            date = pd.to_datetime(snap["date"]).normalize()
            if not isinstance(snap.get("ohlcv"), pd.DataFrame) or snap["ohlcv"].empty:
                if date in ohlcv.index.get_level_values("date"):
                    snap["ohlcv"] = ohlcv.loc[date]
                    if self.verbose:
                        self.logger.debug(f"Added OHLCV to snapshot for {date.date()}")

    # ------------------------------------------------------------------
    # Main day loop  (inside BacktestEngine)
    # ------------------------------------------------------------------

    def _execute_backtest_days(
        self,
        data: List[Dict],
        fm: pd.DataFrame,
        fm_dates: set[pd.Timestamp],
        min_fm_date: pd.Timestamp,
    ) -> pd.DataFrame:
        equity = cash = prev_equity = max_equity = self.initial_equity
        any_trades = positions_built = False

        # 🧷 Bootstrap snapshot on day 0 for correct equity baseline
        first_date = pd.to_datetime(data[0]["date"]).normalize()
        first_prices = data[0]["prices"]

        self.execution.mark_to_market(first_prices)
        self._update_trackers(first_date)
        self.daily_logs.append(
            DailyLog(
                date=first_date,
                prices=first_prices.copy(),
                trades=pd.Series(dtype=float),
                portfolio=self.tracker.get_portfolio().copy(),
                feedback={},
                equity=self.execution.portfolio_value,
                cash=self.execution.current_cash,
            )
        )
        self.logger.info(
            f"[BOOTSTRAP] Initial snapshot on {first_date.date()} | equity=${self.execution.portfolio_value:.2f}"
        )

        with Progress(
            *self._create_progress_columns(),
            console=self.logger.console,
            transient=False,
        ) as prog:
            task = prog.add_task(
                "Backtesting",
                total=len(data),
                date="Starting…",
                equity=equity,
                cash=cash,
                dd=0.0,
                dd_colored="[green]0.00%[/]",
                pnl=0.0,
                pnl_colored="[green]0.00[/]",
            )

            for snap in data:
                date = pd.to_datetime(snap["date"]).normalize()

                try:
                    if not self._should_process_day(date, min_fm_date, fm_dates):
                        continue

                    any_trades, positions_built, _ = self._process_trading_day(
                        snap, date, fm, equity, any_trades, positions_built
                    )

                    equity = getattr(self.execution, "portfolio_value", self.initial_equity)
                    cash = getattr(self.execution, "current_cash", self.initial_equity * 0.5)

                    self.equity_by_date[date] = equity
                    self.cash_by_date[date] = cash

                    daily_pnl, equity, cash, max_equity, dd = self._update_metrics(
                        date, equity, cash, prev_equity, max_equity
                    )
                    prev_equity = equity

                    pnl_color = "green" if daily_pnl >= 0 else "red"
                    dd_color = "green" if dd >= 0 else "red"

                    prog.update(
                        task,
                        advance=1,
                        date=date.strftime("%Y-%m-%d"),
                        equity=equity,
                        cash=cash,
                        dd=dd,
                        dd_colored=f"[{dd_color}]{dd:>6.2%}[/]",
                        pnl=daily_pnl,
                        pnl_colored=f"[{pnl_color}]{daily_pnl:+7.2f}[/]",
                    )

                except Exception as exc:
                    self.logger.error(f"{date.date()} | ⚠️ Exception: {exc}")
                    self.logger.error(traceback.format_exc())

        # Final equity check
        final_prices = data[-1]["prices"]
        final_portfolio = self.tracker.get_portfolio()
        overlap = final_portfolio.index.intersection(final_prices.index)

        recomputed_equity = (final_portfolio[overlap] * final_prices[overlap]).sum()

        self.logger.debug(
            f"[CHECK] execution.portfolio_value = {self.execution.portfolio_value:.2f}"
        )
        self.logger.debug(f"[CHECK] recomputed_equity = {recomputed_equity:.2f} (tracker * prices)")
        self.logger.debug(f"[DEBUG] tracker.index (positions): {list(final_portfolio.index)}")
        self.logger.debug(f"[DEBUG] prices.index (EOD prices): {list(final_prices.index)}")
        self.logger.debug(f"[DEBUG] common symbols: {list(overlap)}")
        for sym in final_portfolio.index:
            price = final_prices.get(sym, None)
            self.logger.debug(
                f"[CHECK] {sym} | weight: {final_portfolio[sym]:.4f} | price: {price}"
            )

        return pd.DataFrame([dl.__dict__ for dl in self.daily_logs]).set_index("date")

    # ------------------------------------------------------------------
    # Progress-bar helpers
    # ------------------------------------------------------------------

    def _create_progress_columns(self) -> List:
        """NautilusTrader-style progress bar with aligned, color-coded metrics."""
        return [
            TextColumn("[bold blue]{task.fields[date]}[/]"),
            BarColumn(),
            TextColumn("[progress.percentage][bold]{task.percentage:>3.0f}%[/]"),
            TextColumn(
                "[dim]Equity:[/] [green]${task.fields[equity]:>8.2f}[/]  "
                "[dim]Cash:[/] [cyan]${task.fields[cash]:>8.2f}[/]  "
                "[dim]DD:[/] {task.fields[dd_colored]}  "
                "[dim]P&L:[/] ${task.fields[pnl_colored]}"
            ),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

    # ------------------------------------------------------------------
    # Metrics & progress helpers
    # ------------------------------------------------------------------
    def _update_metrics(
        self,
        date: pd.Timestamp,
        equity: float,
        cash: float,
        prev_equity: float,
        max_equity: float,
    ) -> Tuple[float, float, float, float, float]:
        """Return daily_pnl, new_equity, new_cash, new_max_equity, drawdown."""
        daily_pnl = equity - prev_equity
        max_equity = max(max_equity, equity)
        drawdown = (equity - max_equity) / max_equity if max_equity else 0.0

        self.daily_pnl[date] = daily_pnl
        self.max_equity = max_equity
        self.drawdown_by_date[date] = drawdown

        # sanity-check cash bounds
        if cash < -equity or cash > 2 * equity:
            cash = equity * 0.5
            if self.verbose:
                self.logger.warning(f"Cash value adjusted for {date.date()}")

        return daily_pnl, equity, cash, max_equity, drawdown

    def _update_progress_bar(
        self,
        prog: Progress,
        task_id,
        date: pd.Timestamp,
        equity: float,
        cash: float,
        dd: float,
        pnl: float,
    ) -> None:
        # 🔍 Debug what the bar is receiving
        actual_equity = getattr(self.execution, "portfolio_value", equity)
        actual_cash = getattr(self.execution, "current_cash", cash)

        self.logger.debug(
            f"[DEBUG] Progress bar update for {date.date()}: "
            f"equity={equity:.2f} | cash={cash:.2f} | "
            f"execution.portfolio_value={actual_equity:.2f} | "
            f"execution.cash={actual_cash:.2f}"
        )

        prog.update(
            task_id,
            advance=1,
            date=date.date(),
            equity=actual_equity,
            cash=actual_cash,
            dd=dd,
            pnl=pnl,
            pnl_prefix="[bold green]" if pnl >= 0 else "[bold red]",
        )

    def _should_process_day(
        self,
        date: pd.Timestamp,
        min_fm_date: pd.Timestamp,
        fm_dates: set[pd.Timestamp],
    ) -> bool:
        """Skip warm-up days and any date missing from feature-matrix."""
        if date < min_fm_date:
            if self.verbose:
                self.logger.debug(f"{date.date()} | 🔄 Warm-up day (no features yet)")
            return False
        if date not in fm_dates:
            if self.verbose:
                self.logger.warning(f"{date.date()} | ⚠️ Date missing from feature matrix, skipping")
            return False
        return True

    def _update_portfolio_values(self, date: pd.Timestamp) -> Tuple[float, float]:
        """
        Update and record end-of-day portfolio equity and cash values.
        Expects the execution model to provide accurate numbers.
        """
        if not hasattr(self.execution, "portfolio_value") or not hasattr(
            self.execution, "current_cash"
        ):
            raise AttributeError(
                "Execution model must expose `portfolio_value` and `current_cash` attributes."
            )

        equity = self.execution.portfolio_value
        cash = self.execution.current_cash

        # Sanity checks
        if not isinstance(equity, (int, float)) or pd.isna(equity):
            raise ValueError(f"Invalid equity value on {date}: {equity}")
        if not isinstance(cash, (int, float)) or pd.isna(cash):
            raise ValueError(f"Invalid cash value on {date}: {cash}")

        # Optional: guard against wild values
        if cash < -equity or cash > 2 * equity:
            self.logger.warning(
                f"⚠️ Cash {cash:.2f} out of expected range on {date}, resetting to 50% of equity."
            )
            cash = 0.5 * equity

        # Save to historical record
        self.equity_by_date[date] = equity
        self.cash_by_date[date] = cash

        return equity, cash

    # ------------------------------------------------------------------
    # Trading-day processing helpers
    # ------------------------------------------------------------------

    def _process_trading_day(
        self,
        snap: Dict,
        date: pd.Timestamp,
        fm: pd.DataFrame,
        equity: float,
        any_trades: bool,
        positions_built: bool,
    ) -> Tuple[bool, bool, Optional[pd.Series]]:
        """Run alpha → portfolio → execution for a single trading day."""

        # 1️⃣ Attach feature vector for the current date
        try:
            snap["feature_vector"] = fm.loc[date]
        except KeyError:
            self.logger.warning(f"{date.date()} | ⚠️ No features found in matrix — skipping day")
            return any_trades, positions_built, None

        # 2️⃣ Generate alpha signals
        signals = self.alpha.predict(snap)
        if signals.empty:
            if self.verbose:
                self.logger.warning(f"{date.date()} | ⚠️ No alpha signals generated")
            return any_trades, positions_built, None

        if self.verbose:
            self.logger.debug(
                f"{date.date()} | Alpha: {len(signals)} | "
                f"Top: {signals.sort_values(ascending=False).head(5).to_dict()}"
            )

        # 3️⃣ Construct portfolio from signals
        snap["capital"] = getattr(self.execution, "portfolio_value", self.initial_equity)
        positions = self.portfolio.construct(signals, snap)
        positions_built |= not positions.empty

        if self.verbose:
            self._log_position_info(positions)

        # 4️⃣ Simulate trades based on positions
        trades = self._simulate_day(date, snap, positions)
        if trades is not None and not trades.empty:
            any_trades = True

        if trades is None or trades.empty:
            # No trades — still mark portfolio and log snapshot
            self.execution.mark_to_market(snap["prices"])
            self._update_trackers(date)

            self.daily_logs.append(
                DailyLog(
                    date=date,
                    prices=snap["prices"].copy(),
                    trades=pd.Series(dtype=float),
                    portfolio=self.tracker.get_portfolio().copy(),
                    feedback={},
                    equity=self.execution.portfolio_value,
                    cash=self.execution.current_cash,
                )
            )

            return any_trades, positions_built, None

        return any_trades, positions_built, trades

    def _simulate_day(
        self, date: pd.Timestamp, snap: Dict, positions: pd.Series
    ) -> Optional[pd.Series]:
        """Run risk, cost, reconciliation, and execution for one day."""
        prices = snap["prices"]
        signals = self.alpha.predict(snap)
        tradable = signals.index.intersection(prices.index)
        if tradable.empty:
            if self.verbose:
                self.logger.warning(f"{date.date()} | No tradable symbols")
            return None

        trades = self._process_signals_to_trades(date, signals.loc[tradable], snap, prices)
        if trades is None or trades.empty:
            return None

        return self._execute_trades(date, trades, prices)

    def _process_signals_to_trades(
        self,
        date: pd.Timestamp,
        signals: pd.Series,
        snap: Dict,
        prices: pd.Series,
    ) -> Optional[pd.Series]:
        """Alpha → risk → cost → portfolio → trades."""
        current = self.tracker.get_portfolio()

        risk_adj = self.risk.apply(signals, current)
        cost_adj = self.cost.adjust(risk_adj, current)

        snap.setdefault("capital", getattr(self.execution, "portfolio_value", self.initial_equity))
        target = self.portfolio.construct(cost_adj, snap)
        trades = reconcile_trades(current, target)

        if trades.empty and self.verbose:
            self.logger.warning(f"{date.date()} | No trades post-reconciliation")

        return trades if not trades.empty else None

    def _execute_trades(
        self, date: pd.Timestamp, trades: pd.Series, prices: pd.Series
    ) -> Optional[pd.Series]:
        """Simulate execution, update state, and log a DailyLog."""
        # 1️⃣ Get available capital
        capital = getattr(self.execution, "portfolio_value", self.initial_equity)

        # 2️⃣ Simulate execution with slippage and impact
        result: TradeResult = simulate_execution(
            trades,
            prices,
            slippage=self.slippage,
            capital=capital,
        )

        # 3️⃣ Apply min holding period filter
        filtered = self.tracker.filter(result.executed, date, self.min_holding)

        # 4️⃣ Log trade feedback
        self.execution.record(filtered, result.feedback)

        # 5️⃣ Get current portfolio, apply trade deltas
        current_portfolio = self.tracker.get_portfolio()
        updated_portfolio = self.execution.update_portfolio(current_portfolio, filtered, capital)

        # 6️⃣ Update tracker with new weights (before MTM!)
        self.tracker.update(updated_portfolio, date)

        # 7️⃣ Revalue portfolio with updated positions
        self.execution.mark_to_market(prices)

        # 8️⃣ Track equity and cash in time series
        self._update_trackers(date)

        # 9️⃣ Feedback for portfolio model
        self.portfolio.feedback_from_execution(result.feedback)

        # 🔟 Log DailyLog snapshot
        self.logger.info(f"{date.date()} | {len(filtered)} trades executed")

        self.daily_logs.append(
            DailyLog(
                date=date,
                prices=prices.copy(),
                trades=filtered.copy(),
                portfolio=updated_portfolio.copy(),
                feedback=result.feedback.copy(),
                equity=self.execution.portfolio_value,
                cash=self.execution.current_cash,
            )
        )

        return filtered

    # ------------------------------------------------------------------
    # Auxiliary log helpers
    # ------------------------------------------------------------------
    def _update_trackers(self, date: pd.Timestamp) -> None:
        self.equity_by_date[date] = getattr(self.execution, "portfolio_value", self.initial_equity)
        self.cash_by_date[date] = getattr(self.execution, "current_cash", self.initial_equity * 0.5)

    def _log_position_info(self, positions: pd.Series) -> None:
        gross = positions.abs().sum() if isinstance(positions, pd.Series) else 0
        self.logger.info(f"✅ Constructed {len(positions)} positions | Gross exposure: {gross:.2f}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _plot_equity_curve(self) -> None:
        from blackbox.utils.plotting import plot_equity_curve

        plot_equity_curve(self.daily_logs, self.config.run_id, self.output_dir)
