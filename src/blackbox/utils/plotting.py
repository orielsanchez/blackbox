from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from blackbox.core.types.types import DailyLog


def plot_equity_curve(
    logs: list[DailyLog],
    run_id: str = "default",
    output_dir: Path = Path(),
    logger: Optional[object] = None,  # Optional RichLogger or print fallback
):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cumulative_equity.png"

    # Extract equity data
    records = [
        {"date": log.date, "equity": log.equity}
        for log in logs
        if hasattr(log, "equity") and log.equity is not None
    ]

    if not records:
        msg = "❌ No valid equity records found. Skipping plot."
        (logger.warning if logger else print)(msg)
        return

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # Calculate returns and drawdowns
    df["cum_return"] = df["equity"] / df["equity"].iloc[0]
    df["rolling_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_return"] / df["rolling_max"] - 1

    if df["cum_return"].isnull().all():
        msg = "❌ Cumulative returns are all NaN. Check equity data."
        (logger.warning if logger else print)(msg)
        return

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["cum_return"], label="Equity Curve", linewidth=2)
    plt.fill_between(df.index, df["drawdown"], 0, color="red", alpha=0.3, label="Drawdown")
    plt.title(f"Equity Curve & Drawdowns — {run_id}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save
    plt.savefig(output_path)
    plt.close()

    msg = f"✅ Equity curve saved to {output_path}"
    (logger.info if logger else print)(msg)
