import json
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
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
        history = history.copy()

        # Ensure datetime index
        if "date" in history.columns:
            history["date"] = pd.to_datetime(history["date"])
            history.set_index("date", inplace=True)
        if not isinstance(history.index, pd.DatetimeIndex):
            raise ValueError("Backtest history must have a DatetimeIndex or 'date' column.")

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
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

        start_date = str(equity.index.min().date())
        end_date = str(equity.index.max().date())

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
            "Start Date": start_date,
            "End Date": end_date,
        }

        # ✅ Add IC metrics if available
        if "ic" in history.columns:
            ic_series = history["ic"].dropna()
            if not ic_series.empty:
                avg_ic = ic_series.mean()
                std_ic = ic_series.std()
                ir = avg_ic / std_ic if std_ic != 0 else 0.0

                metrics.update(
                    {
                        "Avg Information Coefficient": round(avg_ic, 4),
                        "IC StdDev": round(std_ic, 4),
                        "Information Ratio": round(ir, 4),
                    }
                )

                # Export IC timeseries
                ic_series.to_csv("runs/ic_timeseries.csv")

                # Plot IC timeseries
                plt.figure(figsize=(10, 3))
                ic_series.plot(title="Daily Information Coefficient (IC)")
                plt.axhline(0, linestyle="--", color="gray")
                plt.tight_layout()
                plt.savefig("runs/ic_timeseries.png")

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


def load_metrics_for_run(run_id: str, base_dir: Path = Path("backtests")) -> Dict[str, Any]:
    """
    Loads metrics.json from a previous backtest run directory.

    Args:
        run_id (str): Unique ID of the run, e.g. "tune_rsi-14_threshold-0.3"
        base_dir (Path): Where all backtest results are stored.

    Returns:
        dict: Dictionary of metrics like sharpe, return, drawdown, IC, etc.
    """
    metrics_path = base_dir / run_id / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json found at: {metrics_path}")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    return metrics
