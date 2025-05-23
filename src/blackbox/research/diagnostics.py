import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_signal_vs_next_return(
    features: pd.DataFrame,
    signal_column: str,
    close_column: str = "close",
    title: str = "Signal vs Next-Day Return",
    sample_frac: float = 0.05,
):
    """
    Plot scatter of signal vs next-day return to evaluate alpha quality.
    """
    assert {"date", "symbol"} <= set(features.index.names)

    df = features[[signal_column, close_column]].copy()
    df["next_close"] = df.groupby("symbol")[close_column].shift(-1)
    df["next_return"] = df["next_close"] / df[close_column] - 1
    df.dropna(inplace=True)

    df_sample = df.sample(frac=sample_frac, random_state=42)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_sample,
        x=signal_column,
        y="next_return",
        alpha=0.2,
        edgecolor=None,
    )
    sns.regplot(
        data=df_sample,
        x=signal_column,
        y="next_return",
        scatter=False,
        color="red",
        line_kws={"label": "OLS fit"},
    )
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.title(title)
    plt.xlabel(signal_column)
    plt.ylabel("Next-Day Return")
    plt.legend()
    plt.tight_layout()
