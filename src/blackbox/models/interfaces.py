from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class AlphaModel(Protocol):
    name: str

    def generate(self, snapshot: dict) -> pd.Series:
        """
        Generate signals from a daily snapshot.
        Input: snapshot = {
            "date": pd.Timestamp,
            "prices": pd.Series,
            "ohlcv": pd.DataFrame
        }
        Output: pd.Series [symbol → signal]
        """


@runtime_checkable
class RiskModel(Protocol):
    name: str

    def apply(self, signals: pd.Series, portfolio: pd.Series) -> pd.Series:
        """
        Apply risk constraints (e.g., leverage, position limits).
        Output: pd.Series [symbol → risk-adjusted signal]
        """


@runtime_checkable
class TransactionCostModel(Protocol):
    name: str

    def adjust(self, signals: pd.Series, portfolio: pd.Series) -> pd.Series:
        """
        Apply cost-based signal adjustment.
        Output: pd.Series [symbol → cost-adjusted signal]
        """


@runtime_checkable
class PortfolioConstructionModel(Protocol):
    name: str

    def construct(self, adjusted_signals: pd.Series) -> pd.Series:
        """
        Convert adjusted signals into target portfolio weights.
        Output: pd.Series [symbol → weight]
        """

    def feedback_from_execution(self, feedback: dict):
        """
        Optionally respond to trade execution feedback.
        """


@runtime_checkable
class ExecutionModel(Protocol):
    name: str

    def record(self, trades: pd.Series, feedback: dict):
        """
        Track executed trades and metadata.
        """

    def update_portfolio(self, current: pd.Series, trades: pd.Series) -> pd.Series:
        """
        Apply trades to current portfolio to get updated portfolio.
        Output: pd.Series [symbol → new weight]
        """
