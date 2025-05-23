# walkforward_plot.py
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_all_test_curves(
    base_dir: str = "backtests",
    dir_prefixes: tuple[str, ...] = ("test_", "train_"),
) -> pd.DataFrame:
    """
    Collect every equity_curve.csv found directly under folders whose names
    start with any of *dir_prefixes*.

    Parameters
    ----------
    base_dir : str
        Top-level backtest directory.
    dir_prefixes : tuple[str, ...]
        Folder name prefixes to include (order matters for sorting).

    Returns
    -------
    pd.DataFrame
        Concatenated and date-sorted equity curves with an extra 'run_id' column.
    """
    base = Path(base_dir)
    run_dirs = sorted(
        p
        for p in base.iterdir()
        if p.is_dir() and any(p.name.startswith(pref) for pref in dir_prefixes)
    )

    curves: list[pd.DataFrame] = []
    for d in run_dirs:
        curve_path = d / "equity_curve.csv"
        if curve_path.exists():
            df = pd.read_csv(curve_path, parse_dates=["date"]).sort_values("date")
            df["run_id"] = d.name
            curves.append(df)

    if not curves:  # still nothing? tell the user exactly what was searched
        raise FileNotFoundError(
            f"No equity_curve.csv files found under '{base_dir}'. "
            f"Searched prefixes: {dir_prefixes}. "
            f"Verify that your walk-forward runs save equity curves."
        )

    return pd.concat(curves, ignore_index=True).sort_values("date")


def load_spy_benchmark(spy_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.read_csv(spy_path, parse_dates=["date"]).set_index("date")
    df = df.loc[start_date:end_date].copy()
    df["return"] = df["close"].pct_change().fillna(0)
    df["cumulative_return"] = (1 + df["return"]).cumprod()
    df = df.reset_index()[["date", "cumulative_return"]]
    df["cumulative_return"] /= df["cumulative_return"].iloc[0]
    return df


def plot_combined_equity_curve(curves: pd.DataFrame, benchmark: pd.DataFrame):
    curves = curves.sort_values(["date", "run_id"]).drop_duplicates("date", keep="last")
    curves["cumulative_return"] /= curves["cumulative_return"].iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(curves["date"], curves["cumulative_return"], label="Walkforward Strategy")
    plt.plot(
        benchmark["date"],
        benchmark["cumulative_return"],
        label="SPY Benchmark",
        linestyle="--",
    )
    plt.title("📈 Walkforward Equity Curve vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
