from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class RiskModel(ABC):
    @abstractmethod
    def score(
        self, data: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Return risk scores per symbol (e.g., volatility) up to `timestamp`.
        """
        pass
