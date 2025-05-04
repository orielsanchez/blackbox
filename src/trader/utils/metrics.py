from typing import Optional

import numpy as np
import pandas as pd


def calculate_performance(
    equity_curve: pd.Series, trades: Optional[pd.DataFrame] = None
) -> dict:
    """
    Calculate key performance metrics from an equity curve and optional trades.
    Expects:
    - equity_curve: pd.Series indexed by datetime with portfolio value
    - trades: optional pd.DataFrame with columns ['type', 'price', 'quantity']
    """
    if equity_curve.empty:
        return {}

    returns = equity_curve.pct_change().dropna()
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    total_return = end_value / start_value - 1

    index = pd.to_datetime(equity_curve.index)
    num_days = (index[-1] - index[0]).days
    years = num_days / 365.25 if num_days > 0 else 0

    # CAGR
    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else np.nan

    # Daily stats
    daily_return = returns.mean()
    daily_vol = returns.std()
    sharpe_ratio = (
        (daily_return / daily_vol * np.sqrt(252)) if daily_vol > 0 else np.nan
    )

    # Max drawdown
    cumulative_max = equity_curve.cummax()
    drawdown = equity_curve / cumulative_max - 1
    max_drawdown = drawdown.min()

    metrics = {
        "Start Value": round(start_value, 2),
        "End Value": round(end_value, 2),
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}" if pd.notna(cagr) else "N/A",
        "Sharpe Ratio": round(sharpe_ratio, 2) if pd.notna(sharpe_ratio) else "N/A",
        "Volatility": (
            f"{daily_vol * np.sqrt(252):.2%}" if pd.notna(daily_vol) else "N/A"
        ),
        "Max Drawdown": f"{max_drawdown:.2%}" if pd.notna(max_drawdown) else "N/A",
        "Num Days": num_days,
    }

    if trades is not None and not trades.empty:
        trades = trades.copy()
        trades["pnl"] = trades.apply(
            lambda row: row["price"]
            * row["quantity"]
            * (+1 if row["type"] == "SELL" else -1),
            axis=1,
        )
        trade_pnls = trades.groupby("symbol")["pnl"].sum()
        wins = trade_pnls[trade_pnls > 0]
        win_rate = len(wins) / len(trade_pnls) if len(trade_pnls) > 0 else 0
        metrics.update(
            {
                "Total Trades": len(trades),
                "Winning Trades": len(wins),
                "Win Rate": f"{win_rate:.2%}",
                "Avg Trade PnL": f"{trades['pnl'].mean():.2f}",
            }
        )

    return metrics
