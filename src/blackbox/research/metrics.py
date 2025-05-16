from typing import Dict, Union

import numpy as np
import pandas as pd


class PerformanceMetrics:
    def __init__(self, initial_value: float = 1_000_000, risk_free_rate: float = 0.0):
        self.initial_value = initial_value
        self.risk_free_rate = risk_free_rate

    def compute(
        self, history: pd.DataFrame, return_equity: bool = False
    ) -> Union[Dict[str, float | str], tuple[Dict[str, float | str], pd.Series]]:
        """
        Compute key performance metrics from backtest history.

        Parameters:
        - history: pd.DataFrame with 'portfolio' and 'prices' columns.
        - return_equity: If True, also return the equity curve.

        Returns:
        - metrics dict (and optionally equity curve)
        """
        equity = self._compute_equity_curve(history)
        returns = equity.pct_change().fillna(0)

        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (
            (annual_return - self.risk_free_rate) / annual_volatility
            if annual_volatility != 0
            else 0
        )

        sortino_ratio = self._sortino_ratio(returns)
        max_drawdown = self._max_drawdown(equity)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

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
            "Start Date": str(equity.index[0].date()),
            "End Date": str(equity.index[-1].date()),
        }

        if return_equity:
            return metrics, equity
        return metrics

    def _compute_equity_curve(self, history: pd.DataFrame) -> pd.Series:
        if "equity" not in history.columns:
            raise ValueError("Missing 'equity' column in backtest history")

        equity_curve = history["equity"].astype(float).copy()
        equity_curve.name = "NetAssetValue"
        return equity_curve

    def _max_drawdown(self, series: pd.Series) -> float:
        peak = series.expanding(min_periods=1).max()
        drawdown = (series - peak) / peak
        return drawdown.min()

    def _sortino_ratio(self, returns: pd.Series) -> float:
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        if downside_vol == 0:
            return 0.0
        mean_excess_return = returns.mean() * 252 - self.risk_free_rate
        return mean_excess_return / downside_vol
