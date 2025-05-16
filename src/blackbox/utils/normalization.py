import numpy as np
import pandas as pd


class SignalNormalizer:
    @staticmethod
    def zscore(signal: pd.Series) -> pd.Series:
        return (signal - signal.mean()) / signal.std(ddof=0)

    @staticmethod
    def minmax(signal: pd.Series) -> pd.Series:
        min_val, max_val = signal.min(), signal.max()
        return (signal - min_val) / (max_val - min_val) if max_val > min_val else signal

    @staticmethod
    def rank(signal: pd.Series) -> pd.Series:
        return signal.rank(method="average") / len(signal)

    @staticmethod
    def softmax(signal: pd.Series) -> pd.Series:
        exp_signal = np.exp(signal - signal.max())  # for numerical stability
        return exp_signal / exp_signal.sum()

    @staticmethod
    def winsorized_zscore(signal: pd.Series, lower=0.05, upper=0.95) -> pd.Series:
        clipped = signal.clip(signal.quantile(lower), signal.quantile(upper))
        return (clipped - clipped.mean()) / clipped.std(ddof=0)
