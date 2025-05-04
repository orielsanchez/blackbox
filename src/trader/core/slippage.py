from abc import ABC, abstractmethod

import pandas as pd


class SlippageModel(ABC):
    @abstractmethod
    def apply(self, orders: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        pass
