from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class PositionMeta:
    entry_date: pd.Timestamp
    weight: float


class PositionTracker:
    def __init__(self):
        self.positions: Dict[str, PositionMeta] = {}

    def get_portfolio(self) -> pd.Series:
        return pd.Series(
            {
                symbol: meta.weight
                for symbol, meta in self.positions.items()
                if abs(meta.weight) > 1e-6
            }
        ).sort_index()

    def can_trade(self, symbol: str, date: pd.Timestamp, min_holding: int) -> bool:
        if symbol not in self.positions:
            return True
        held_days = (date - self.positions[symbol].entry_date).days
        return held_days >= min_holding

    def filter(
        self, trades: pd.Series, date: pd.Timestamp, min_holding: int
    ) -> pd.Series:
        filtered = {
            symbol: weight
            for symbol, weight in trades.items()
            if weight >= 0 or self.can_trade(symbol, date, min_holding)
        }
        return pd.Series(filtered).sort_index()

    def update(self, updated_portfolio: pd.Series, date: pd.Timestamp):
        for symbol, weight in updated_portfolio.items():
            prev = self.positions.get(symbol)
            if prev is None or abs(prev.weight) < 1e-6:
                self.positions[symbol] = PositionMeta(entry_date=date, weight=weight)
            else:
                self.positions[symbol].weight = weight

        for symbol in list(self.positions):
            if abs(updated_portfolio.get(symbol, 0.0)) < 1e-6:
                del self.positions[symbol]
