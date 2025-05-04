from abc import ABC, abstractmethod

import pandas as pd


class ExecutionModel(ABC):
    @abstractmethod
    def execute_orders(
        self,
        target_positions: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        pass
