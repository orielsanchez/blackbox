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
        self,
        history: pd.DataFrame,
        return_equity: bool = False,
        output_dir: Path = Path("runs"),
    ) -> Union[Dict[str, float | str], tuple[Dict[str, float | str], pd.Series]]:
        """
        Compute standard performance metrics from backtest history.

        Args:
            history: DataFrame with 'equity' column (and optionally 'ic').
            return_equity: Whether to return equity curve along with metrics.
            output_dir: Where to save optional plots or IC series.

        Returns:
            metrics dict (and equity Series if return_equity=True)
        """
        history = history.copy()

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

        metrics = {
            "Start Equity": round(equity.iloc[0], 2),
            "End Equity": round(equity.iloc[-1], 2),
            "Total Return (%)": round(total_return * 100, 2),
            "Annualized Return (%)": round(annual_return * 100, 2),
            "Annualized Volatility (%)": round(annual_volatility * 100, 2),
            "Sharpe Ratio": round(sharpe_ratio, 3),
            "Sortino Ratio": round(self._sortino_ratio(returns), 3),
            "Max Drawdown (%)": round(self._max_drawdown(equity) * 100, 2),
            "Calmar Ratio": round(
                (
                    annual_return / abs(self._max_drawdown(equity))
                    if self._max_drawdown(equity) != 0
                    else np.nan
                ),
                3,
            ),
            "R squared": round(self._compute_r_squared(equity), 4),
            "Start Date": str(equity.index.min().date()),
            "End Date": str(equity.index.max().date()),
        }

        if "ic" in history.columns:
            self._add_ic_metrics(history["ic"], metrics, output_dir)

        return (metrics, equity) if return_equity else metrics

    def _compute_equity_curve(self, history: pd.DataFrame) -> pd.Series:
        if "equity" not in history.columns:
            raise ValueError("Missing required column 'equity' in backtest history.")
        return history["equity"].astype(float).rename("NetAssetValue")

    def _max_drawdown(self, series: pd.Series) -> float:
        peak = series.expanding(min_periods=1).max()
        return ((series - peak) / peak).min()

    def _sortino_ratio(self, returns: pd.Series) -> float:
        downside = returns[returns < 0]
        downside_vol = downside.std() * np.sqrt(252)
        if downside_vol == 0:
            return 0.0
        excess = returns.mean() * 252 - self.risk_free_rate
        return excess / downside_vol

    def _compute_r_squared(self, equity: pd.Series) -> float:
        X = np.arange(len(equity)).reshape(-1, 1)
        y = equity.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return float(model.score(X, y))

    def _add_ic_metrics(
        self, ic_series: pd.Series, metrics: Dict[str, Any], output_dir: Path
    ) -> None:
        ic_series = ic_series.dropna()
        if ic_series.empty:
            return

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

        output_dir.mkdir(parents=True, exist_ok=True)
        ic_series.to_csv(output_dir / "ic_timeseries.csv")

        plt.figure(figsize=(10, 3))
        ic_series.plot(title="Daily Information Coefficient (IC)")
        plt.axhline(0, linestyle="--", color="gray")
        plt.tight_layout()
        plt.savefig(output_dir / "ic_timeseries.png")
        plt.close()


def load_metrics_for_run(
    run_id: str,
    base_dir: Path = Path("backtests"),
) -> Dict[str, Any]:
    """
    Load performance metrics JSON for a completed backtest run.

    Args:
        run_id: ID of the backtest run
        base_dir: Directory containing backtest output folders

    Returns:
        Dictionary of metrics
    """
    metrics_path = base_dir / run_id / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json found at: {metrics_path}")

    with open(metrics_path, "r") as f:
        data = json.load(f)

    return data["metrics"]
