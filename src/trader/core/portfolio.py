from abc import ABC, abstractmethod

import pandas as pd


class PortfolioConstructionModel(ABC):
    @abstractmethod
    def allocate(
        self,
        alpha_scores: pd.DataFrame,
        risk_scores: pd.DataFrame,
        cost_scores: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        pass
