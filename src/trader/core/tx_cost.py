from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class TransactionCostModel(ABC):
    @abstractmethod
    def estimate(
        self, trades: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        pass
