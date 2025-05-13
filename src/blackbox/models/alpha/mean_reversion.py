from typing import Dict

import pandas as pd

from blackbox.models.interfaces import AlphaModel


class MeanReversionAlpha(AlphaModel):
    name = "mean_reversion"
    """
    Time-series mean reversion strategy.
    Generates long/short signals based on Z-score of close price vs. rolling mean.
    """

    def __init__(self, window: int = 20, threshold: float = 1.5):
        self.window = window
        self.threshold = threshold

    def generate(self, snapshot: Dict) -> pd.Series:
        """
        Args:
            snapshot (Dict): Should contain 'ohlcv': Dict[symbol → pd.DataFrame]

        Returns:
            pd.Series: symbol → alpha signal in [-1, 1]
        """
        ohlcv: Dict[str, pd.DataFrame] = snapshot.get("ohlcv", {})
        signals = {}

        for symbol, df in ohlcv.items():
            if not isinstance(df, pd.DataFrame):
                continue  # malformed entry

            if "close" not in df.columns or len(df) < self.window:
                continue

            close = df["close"].iloc[-self.window :]
            if close.isna().any():
                continue  # skip if recent prices are missing

            mean = close.mean()
            std = close.std()
            if std == 0 or pd.isna(std):
                continue  # avoid divide-by-zero

            zscore = (close.iloc[-1] - mean) / std

            if zscore < -self.threshold:
                signals[symbol] = min(1.0, abs(zscore))  # long
            elif zscore > self.threshold:
                signals[symbol] = -min(1.0, abs(zscore))  # short
            else:
                signals[symbol] = 0.0

        alpha = pd.Series(signals).fillna(0.0)

        # Normalize weights (optional)
        if alpha.abs().sum() > 1.0:
            alpha = alpha / alpha.abs().sum()

        return alpha
