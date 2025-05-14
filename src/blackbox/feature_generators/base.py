from abc import ABC, abstractmethod
from typing import Dict, Type

import pandas as pd

# Registry for all feature generators
feature_registry: Dict[str, Type["BaseFeatureGenerator"]] = {}


def register_feature(name: str):
    def decorator(cls):
        feature_registry[name] = cls
        return cls

    return decorator


class BaseFeatureGenerator(ABC):
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from input OHLCV data.

        Parameters:
            data (pd.DataFrame): MultiIndex [date, symbol] DataFrame with OHLCV

        Returns:
            pd.DataFrame: DataFrame with same index and one or more feature columns
        """
        pass
