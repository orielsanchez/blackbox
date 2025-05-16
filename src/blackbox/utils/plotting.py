import os

import matplotlib.pyplot as plt
import pandas as pd

from blackbox.core.execution_loop import DailyLog

# from pathlib import Path


def plot_equity_curve(
    logs: list[DailyLog], run_id: str = "default", output_dir: str = "results"
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(
        [{"date": log.date, "portfolio_value": log.portfolio.sum()} for log in logs]
    )
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    df["cum_return"] = df["portfolio_value"] / df["portfolio_value"].iloc[0]
    df["rolling_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_return"] / df["rolling_max"] - 1

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["cum_return"], label="Equity Curve", linewidth=2)
    plt.fill_between(
        df.index, df["drawdown"], 0, color="red", alpha=0.3, label="Drawdown"
    )
    plt.title(f"Equity Curve & Drawdowns — {run_id}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"equity_{run_id}.png")
    plt.savefig(output_path)
    plt.close()
