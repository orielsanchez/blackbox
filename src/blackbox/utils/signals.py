from typing import Optional

import pandas as pd
from scipy.stats import rankdata, zscore


def normalize_signal(
    signal: pd.Series,
    method: str = "zscore_cs",
    window: int = 20,
    cap: float = 3.0,
    volatility: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Normalize a signal using various methods.

    Args:
        signal: Raw signal series (MultiIndex: [date, symbol]).
        method: One of:
            - "zscore_cs"     = cross-sectional z-score
            - "percentile_cs" = cross-sectional percentile rank
            - "zscore_ts"     = time-series z-score per symbol
            - "vol_adj"       = divide by rolling std or ATR (pass volatility)
        window: Lookback window for time-series or volatility normalization.
        cap: Clip output to [-cap, cap].
        volatility: Optional precomputed volatility series (same index as signal).

    Returns:
        Normalized and capped pd.Series with same index as input.
    """
    if not isinstance(signal.index, pd.MultiIndex):
        raise ValueError("Signal must have MultiIndex [date, symbol]")

    if method == "zscore_cs":
        normalized = signal.groupby(level="date").transform(
            lambda x: (zscore(x, ddof=0) if x.std(ddof=0) > 0 else pd.Series(0.0, index=x.index))
        )

    elif method == "percentile_cs":
        normalized = signal.groupby(level="date").transform(
            lambda x: (
                pd.Series((rankdata(x, method="average") - 1) / (len(x) - 1), index=x.index)
                if len(x) > 1
                else pd.Series(0.0, index=x.index)
            )
        )

    elif method == "zscore_ts":
        normalized = signal.groupby(level="symbol").transform(
            lambda x: (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-8)
        )

    elif method == "vol_adj":
        if volatility is None:
            volatility = signal.groupby(level="symbol").transform(lambda x: x.rolling(window).std())

        assert isinstance(volatility, pd.Series)
        normalized = signal / (volatility + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return pd.Series(normalized, index=signal.index).clip(lower=-cap, upper=cap)
