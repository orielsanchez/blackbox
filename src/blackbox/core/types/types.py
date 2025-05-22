from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class DailyLog:
    date: pd.Timestamp
    prices: pd.Series
    trades: pd.Series
    portfolio: pd.Series
    feedback: dict[str, dict[str, object]] = field(default_factory=dict)
    equity: float = 0.0
    cash: float = 0.0
    pnl: float = 0.0
    drawdown: float = 0.0
    ic: Optional[float] = None


@dataclass
class OHLCVSnapshot:
    date: pd.Timestamp
    close: pd.Series
    open: pd.Series
    high: pd.Series
    low: pd.Series
    volume: pd.Series
    next_close: Optional[pd.Series] = None
    other: dict[str, pd.Series] = field(default_factory=dict)


@dataclass
class PortfolioTarget:
    date: pd.Timestamp
    weights: pd.Series
    capital: float
    signals: Optional[pd.Series] = None
    execution_method: str = "market"
    limit_prices: Optional[pd.Series] = None


@dataclass
class TradeResult:
    executed: pd.Series
    fill_prices: pd.Series
    feedback: dict[str, dict[str, object]]
