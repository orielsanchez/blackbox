from abc import ABC, abstractmethod
import pandas as pd


class AlphaModel(ABC):
    @abstractmethod
    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
