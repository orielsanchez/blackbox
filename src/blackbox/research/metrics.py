from typing import Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class PerformanceMetrics:
    def __init__(self, initial_value: float = 1_000_000, risk_free_rate: float = 0.0):
        self.initial_value = initial_value
        self.risk_free_rate = risk_free_rate

    def compute_metrics(
        self, history: pd.DataFrame, return_equity: bool = False
    ) -> Union[dict[str, float | str], tuple[dict[str, float | str], pd.Series]]:
        """
        Compute key performance metrics from backtest history.

        Parameters:
            history (pd.DataFrame): Must contain an 'equity' column and datetime index or 'date' column.
            return_equity (bool): If True, also return the equity curve as a Series.

        Returns:
            dict or (dict, Series): Metrics dictionary, and optionally the equity curve.
        """
        history = history.copy()

        # Ensure datetime index
        if "date" in history.columns:
            history["date"] = pd.to_datetime(history["date"])
            history.set_index("date", inplace=True)
        if not isinstance(history.index, pd.DatetimeIndex):
            raise ValueError(
                "Backtest history must have a DatetimeIndex or 'date' column."
            )

        equity = self._compute_equity_curve(history)
        returns = equity.pct_change().fillna(0)

        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (
            (annual_return - self.risk_free_rate) / annual_volatility
            if annual_volatility != 0
            else 0.0
        )

        r_squared = self._compute_r_squared(equity)
        sortino_ratio = self._sortino_ratio(returns)
        max_drawdown = self._max_drawdown(equity)
        calmar_ratio = (
            annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
        )

        start_date = str(pd.Timestamp(equity.index.min()).date())
        end_date = str(pd.Timestamp(equity.index.max()).date())
        metrics = {
            "Start Equity": round(equity.iloc[0], 2),
            "End Equity": round(equity.iloc[-1], 2),
            "Total Return (%)": round(total_return * 100, 2),
            "Annualized Return (%)": round(annual_return * 100, 2),
            "Annualized Volatility (%)": round(annual_volatility * 100, 2),
            "Sharpe Ratio": round(sharpe_ratio, 3),
            "Sortino Ratio": round(sortino_ratio, 3),
            "Max Drawdown (%)": round(max_drawdown * 100, 2),
            "Calmar Ratio": round(calmar_ratio, 3),
            "R squared": round(r_squared, 4),
            "Start Date": str(start_date),
            "End Date": str(end_date),
        }

        return (metrics, equity) if return_equity else metrics

    def _compute_equity_curve(self, history: pd.DataFrame) -> pd.Series:
        if "equity" not in history.columns:
            raise ValueError("Missing required column 'equity' in backtest history.")
        equity = history["equity"].astype(float)
        equity.name = "NetAssetValue"
        return equity

    def _max_drawdown(self, series: pd.Series) -> float:
        peak = series.expanding(min_periods=1).max()
        drawdown = (series - peak) / peak
        return drawdown.min()

    def _sortino_ratio(self, returns: pd.Series) -> float:
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        if downside_vol == 0:
            return 0.0
        excess_return = returns.mean() * 252 - self.risk_free_rate
        return excess_return / downside_vol

    def _compute_r_squared(self, equity: pd.Series) -> float:
        X = np.arange(len(equity)).reshape(-1, 1)
        y = equity.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return float(model.score(X, y))
