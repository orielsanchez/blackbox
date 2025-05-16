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
    Tracks positions with entry dates and weights to:
    - Enforce minimum holding periods
    - Build the current portfolio from state
    """

    def __init__(self):
        self.positions: Dict[str, PositionMeta] = {}

    def get_portfolio(self) -> pd.Series:
        """
        Returns current portfolio as a Series of weights,
        filtered to exclude near-zero positions.
        """
        return pd.Series(
            {
                symbol: meta.weight
                for symbol, meta in self.positions.items()
                if not isclose(meta.weight, 0.0, abs_tol=1e-6)
            }
        ).sort_index()

    def can_trade(
        self, symbol: str, current_date: pd.Timestamp, min_holding: int
    ) -> bool:
        """
        Returns True if the symbol can be traded today,
        based on whether it's held long enough (min_holding in days).
        """
        meta = self.positions.get(symbol)
        if meta is None:
            return True  # Not held yet
        days_held = (current_date - meta.entry_date).days
        return days_held >= min_holding

    def filter(
        self, trades: pd.Series, date: pd.Timestamp, min_holding: int
    ) -> pd.Series:
        """
        Filters trades based on holding period logic:
        - Long positions (weight ≥ 0) are always allowed
        - Short positions (weight < 0) require min holding period
        """
        filtered = {
            symbol: weight
            for symbol, weight in trades.items()
            if weight >= 0 or self.can_trade(symbol, date, min_holding)
        }
        return pd.Series(filtered).sort_index()

    def update(self, updated_portfolio: pd.Series, date: pd.Timestamp):
        """
        Updates internal state with a new portfolio:
        - Adds new positions with today's entry date
        - Updates weights for existing ones
        - Removes any positions with weight ≈ 0
        """
        # Update or create positions
        for symbol, weight in updated_portfolio.items():
            existing = self.positions.get(symbol)
            if existing is None or isclose(existing.weight, 0.0, abs_tol=1e-6):
                self.positions[symbol] = PositionMeta(entry_date=date, weight=weight)
            else:
                existing.weight = weight

        # Remove exited positions
        to_remove = [
            symbol
            for symbol, meta in self.positions.items()
            if isclose(updated_portfolio.get(symbol, 0.0), 0.0, abs_tol=1e-6)
        ]
        for symbol in to_remove:
            del self.positions[symbol]
