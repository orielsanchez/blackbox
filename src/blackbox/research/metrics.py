from typing import Dict

import numpy as np
import pandas as pd


class PerformanceMetrics:
    def __init__(self, initial_value: float = 1_000_000):
        self.initial_value = initial_value

    def compute(self, history: pd.DataFrame) -> Dict[str, float | str]:
        """
        Computes performance metrics from backtest history.

        Assumes history is a DataFrame with:
        - index: datetime (or 'date' column)
        - 'portfolio': pd.Series
        - 'prices': pd.Series
        """
        equity = self._compute_equity_curve(history)

        returns = equity.pct_change().fillna(0)
        cumulative = equity.iloc[-1] / equity.iloc[0] - 1
        annual_return = (1 + cumulative) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol != 0 else 0
        max_dd = self._max_drawdown(equity)

        return {
            "Total Return (%)": round(cumulative * 100, 2),
            "Annualized Return (%)": round(annual_return * 100, 2),
            "Annualized Volatility (%)": round(annual_vol * 100, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Max Drawdown (%)": round(max_dd * 100, 2),
            "Start Date": str(equity.index[0].date()),
            "End Date": str(equity.index[-1].date()),
        }

    def _compute_equity_curve(self, history: pd.DataFrame) -> pd.Series:
        """
        Computes NAV over time using portfolio weights and daily returns.

        Assumes:
        - 'portfolio': pd.Series of weights
        - 'prices': pd.Series of close prices
        """
        # Build a DataFrame of all prices
        price_df = pd.DataFrame([row["prices"] for _, row in history.iterrows()])
        price_df.index = history.index
        price_returns = price_df.pct_change(fill_method=None).fillna(0)

        # Compute daily portfolio return = sum(weights * price returns)
        returns = []
        for i, row in history.iterrows():
            weights = row["portfolio"]
            if i == history.index[0]:
                returns.append(0.0)  # no return on first day
                continue
            ret = (weights * price_returns.loc[i]).sum()
            returns.append(ret)

        # Compute NAV from returns
        equity_curve = pd.Series(returns, index=history.index).add(1).cumprod()
        equity_curve *= self.initial_value
        equity_curve.name = "NetAssetValue"
        return equity_curve

    def _max_drawdown(self, series: pd.Series) -> float:
        peak = series.expanding(min_periods=1).max()
        drawdown = (series - peak) / peak
        return drawdown.min()
