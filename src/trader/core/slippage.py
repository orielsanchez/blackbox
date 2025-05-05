from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class SlippageModel(ABC):
    @abstractmethod
    def score(
        self, trades: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        pass
