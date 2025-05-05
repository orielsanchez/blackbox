from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class PortfolioConstructionModel(ABC):
    @abstractmethod
    def allocate(
        self,
        alpha_df: pd.DataFrame,
        risk_df: pd.DataFrame,
        tx_df: pd.DataFrame,
        slippage_df: pd.DataFrame,
        price_df: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        """
        Combine alpha, risk, cost, and price data to construct a portfolio.

        Parameters:
            alpha_df: DataFrame with ['symbol', 'alpha_score']
            risk_df: DataFrame with ['symbol', 'risk_score']
            tx_df: DataFrame with ['symbol', 'tx_cost']
            slippage_df: DataFrame with ['symbol', 'slippage']
            price_df: DataFrame with ['symbol', 'price']
            capital: Total capital to allocate

        Returns:
            DataFrame with ['symbol', 'weight', 'target_value', 'shares']
        """
        pass
