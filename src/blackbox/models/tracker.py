from dataclasses import dataclass
from math import isclose
from typing import Dict

import pandas as pd


@dataclass
class PositionMeta:
    entry_date: pd.Timestamp
    weight: float


class PositionTracker:
    """
    Tracks position entry dates and weights to enforce minimum holding periods and build portfolios.
    """

    def __init__(self):
        self.positions: Dict[str, PositionMeta] = {}

    def get_portfolio(self) -> pd.Series:
        """
        Returns the current portfolio as a Series of weights (filtered for non-zero positions).
        """
        return pd.Series(
            {
                symbol: meta.weight
                for symbol, meta in self.positions.items()
                if not isclose(meta.weight, 0.0, abs_tol=1e-6)
            }
        ).sort_index()

    def can_trade(self, symbol: str, date: pd.Timestamp, min_holding: int) -> bool:
        """
        Checks if a symbol can be traded based on its holding period.
        """
        if symbol not in self.positions:
            return True
        held_days = (date - self.positions[symbol].entry_date).days
        return held_days >= min_holding

    def filter(self, trades: pd.Series, date: pd.Timestamp, min_holding: int) -> pd.Series:
        """
        Filters trades to respect the minimum holding period for existing positions.

        - Longs (weight >= 0) are always allowed.
        - Shorts (weight < 0) are only allowed if held long enough.
        """
        filtered = {
            symbol: weight
            for symbol, weight in trades.items()
            if weight >= 0 or self.can_trade(symbol, date, min_holding)
        }
        return pd.Series(filtered).sort_index()

    def update(self, updated_portfolio: pd.Series, date: pd.Timestamp):
        """
        Updates position metadata for the current date.
        - Adds new entries or updates weights.
        - Removes exited positions (zero weight).
        """
        for symbol, weight in updated_portfolio.items():
            prev = self.positions.get(symbol)
            if prev is None or isclose(prev.weight, 0.0, abs_tol=1e-6):
                # New or reopened position
                self.positions[symbol] = PositionMeta(entry_date=date, weight=weight)
            else:
                # Update existing weight
                self.positions[symbol].weight = weight

        # Remove exited positions
        for symbol in list(self.positions):
            if isclose(updated_portfolio.get(symbol, 0.0), 0.0, abs_tol=1e-6):
                del self.positions[symbol]
