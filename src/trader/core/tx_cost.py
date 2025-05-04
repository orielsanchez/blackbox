from abc import ABC, abstractmethod

import pandas as pd


class TransactionCostModel(ABC):
    @abstractmethod
    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
