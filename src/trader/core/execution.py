from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class ExecutionModel(ABC):
    @abstractmethod
    def execute(
        self,
        orders: pd.DataFrame,
        market_data: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
        cash: float = float("inf"),
    ) -> list[dict]:
        """
        Simulate execution of orders using market data and return a list of fills.
        Each fill is a dict with keys like: symbol, side, quantity, fill_price, slippage, timestamp.
        """
        pass
