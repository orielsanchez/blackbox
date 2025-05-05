from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def plot_log_equity_with_regression(equity_curve: pd.Series, title="Log Equity Curve"):
    if equity_curve.empty:
        print("⚠️ Cannot plot: equity curve is empty.")
        return

    equity_curve = equity_curve.dropna()
    y_raw = equity_curve.values
    x = np.arange(len(y_raw)).reshape(-1, 1)

    valid = y_raw > 0
    x = x[valid]
    y = np.log(y_raw[valid])

    if len(x) < 2:
        print("⚠️ Not enough valid points for regression.")
        return

    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="Log(Equity)", color="blue")
    plt.plot(x, y_pred, label="Linear Fit", linestyle="--", color="red")
    plt.title(title + f" (R² = {model.score(x, y):.4f})")
    plt.xlabel("Time Index")
    plt.ylabel("Log Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_performance(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.0,
) -> dict:
    if equity_curve.empty:
        return {}

    equity_curve = equity_curve.sort_index()
    returns = equity_curve.pct_change().dropna()
    log_returns = np.log1p(returns)

    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    total_return = end_value / start_value - 1

    index = pd.to_datetime(equity_curve.index)
    num_days = (index[-1] - index[0]).days
    years = num_days / 365.25 if num_days > 0 else 0

    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else np.nan

    daily_return = returns.mean()
    daily_vol = returns.std()
    excess_daily_return = daily_return - (risk_free_rate / 252)
    sharpe_ratio = (
        (excess_daily_return / daily_vol * np.sqrt(252)) if daily_vol > 0 else np.nan
    )

    cumulative_max = equity_curve.cummax()
    drawdown = equity_curve / cumulative_max - 1
    max_drawdown = drawdown.min()
    drawdown_duration = (
        (drawdown < 0).astype(int).groupby((drawdown >= 0).cumsum()).sum().max()
    )

    x = np.arange(len(equity_curve)).reshape(-1, 1)
    y = np.log(equity_curve.values.reshape(-1, 1))
    r_squared = LinearRegression().fit(x, y).score(x, y)

    metrics = {
        "Start Value": round(start_value, 2),
        "End Value": round(end_value, 2),
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}" if pd.notna(cagr) else "N/A",
        "Annual Volatility": (
            f"{daily_vol * np.sqrt(252):.2%}" if pd.notna(daily_vol) else "N/A"
        ),
        "Sharpe Ratio": round(sharpe_ratio, 2) if pd.notna(sharpe_ratio) else "N/A",
        "Max Drawdown": f"{max_drawdown:.2%}" if pd.notna(max_drawdown) else "N/A",
        "Drawdown Duration (days)": (
            int(drawdown_duration) if pd.notna(drawdown_duration) else "N/A"
        ),
        "R^2 (log curve)": round(r_squared, 4),
        "Num Days": num_days,
    }

    if trades is not None and not trades.empty:
        trades = trades.copy()
        side_map = trades["side"].map({"sell": 1, "buy": -1})
        side_map.fillna(0, inplace=True)

        trades["impact"] = trades["fill_price"] * trades["quantity"] * side_map
        wins = trades["impact"] > 0
        win_rate = wins.mean() if not wins.empty else np.nan

        metrics.update(
            {
                "Total Trades": len(trades),
                "Winning Trades": int(wins.sum()),
                "Win Rate": f"{win_rate:.2%}" if pd.notna(win_rate) else "N/A",
                "Avg Trade Impact": f"{trades['impact'].mean():.2f}",
            }
        )

    return metrics


def print_performance_summary(metrics: dict) -> None:
    if not metrics:
        print("\n📉 No performance metrics to display.")
        return

    print("\n📊 Backtest Performance Summary")
    print("======================================")
    for key, value in metrics.items():
        print(f"{key:<30} {value}")
    print("======================================")
