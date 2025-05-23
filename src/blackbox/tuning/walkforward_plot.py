# walkforward_plot.py
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _derive_cumulative_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the frame has a ``cumulative_return`` column.

    Accepts either:
    • equity_curve.csv           ─ columns: [date, equity]
    • pre-derived curve          ─ columns: [date, cumulative_return]
    • logs.csv fallback          ─ columns: [date, equity, …]

    Returns *df* with a cumulative_return column.
    """
    if "cumulative_return" in df.columns:
        return df

    if "equity" not in df.columns:
        raise ValueError(
            "Curve file must contain either 'cumulative_return' or 'equity' column."
        )

    df = df.copy()
    # df["cumulative_return"] = df["equity"] / df["equity"].iloc[0]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def load_all_test_curves(
    base_dir: Path = Path("backtests"),
    dir_prefixes: Tuple[str, ...] = ("test_", "train_"),
) -> pd.DataFrame:
    """
    Scan *base_dir* for run folders that start with any prefix in *dir_prefixes*,
    then pull an equity curve from each folder.

    Priority inside each run folder:
    1. equity_curve.csv  (preferred)
    2. logs.csv fallback (must contain 'date' and 'equity')
    """
    base = Path(base_dir)
    run_dirs: List[Path] = sorted(
        p
        for p in base.rglob("*")
        if p.is_dir() and any(p.name.startswith(pref) for pref in dir_prefixes)
    )

    curves: List[pd.DataFrame] = []
    for d in run_dirs:
        # ────────────────── Preferred: equity_curve.csv ──────────────────
        curve_path = d / "equity_curve.csv"
        if curve_path.exists():
            df = pd.read_csv(curve_path, parse_dates=["date"]).sort_values("date")
            df = _derive_cumulative_return(df)
            df["run_id"] = d.name
            curves.append(df)
            continue  # next run_dir

        # ────────────────── Fallback: logs.csv ───────────────────────────
        log_path = d / "logs.csv"
        if log_path.exists():
            log = pd.read_csv(log_path, parse_dates=["date"]).sort_values("date")
            if {"date", "equity"}.issubset(log.columns):
                log = log[["date", "equity"]]
                log = _derive_cumulative_return(log)
                log["run_id"] = d.name
                curves.append(log)

    if not curves:
        raise FileNotFoundError(
            f"No equity data found under '{base_dir}'. "
            f"Searched prefixes: {dir_prefixes}. "
            "Ensure each run folder contains either equity_curve.csv or logs.csv "
            "with 'date' and 'equity' columns."
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


def plot_combined_equity_curve(
    curves: pd.DataFrame, benchmark: pd.DataFrame, session_root: Path
) -> None:
    curves = (
        curves.sort_values(["date", "run_id"])
        .drop_duplicates("date", keep="last")
        .copy()
    )
    # Normalise strategy curve so both start at 1
    # curves["cumulative_return"] /= curves["cumulative_return"].iloc[0]
    curves["cumulative_return"] = curves["equity"] / curves["equity"].iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(curves["date"], curves["cumulative_return"], label="Walk-forward Strategy")
    plt.plot(
        benchmark["date"],
        benchmark["cumulative_return"],
        label="SPY Benchmark",
        linestyle="--",
    )
    plt.title("📈 Walk-forward Equity Curve vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    outfile = session_root / "walkforward_equity_curve.png"
    plt.savefig(outfile, dpi=150)
