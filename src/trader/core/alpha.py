from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class AlphaModel(ABC):
    @abstractmethod
    def score(
        self, data: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Return alpha scores per symbol using historical data up to `timestamp`.
        """
        pass
