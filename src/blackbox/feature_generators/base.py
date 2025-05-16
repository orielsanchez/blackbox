import time
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Type

import pandas as pd

from blackbox.utils.context import get_logger

# === Global registry of available feature generators ===
feature_registry: Dict[str, Type["BaseFeatureGenerator"]] = {}


def register_feature(name: str):
    """
    Decorator to register a feature generator class by name.
    Usage:
        @register_feature("zscore_price")
        class ZScorePriceFeature(...):
            ...
    """

    def decorator(cls):
        feature_registry[name] = cls
        return cls

    return decorator


class BaseFeatureGenerator(ABC):
    """
    Abstract base class for all feature generators.
    Subclasses must implement `generate()`.
    """

    def __init__(self):
        self.logger = get_logger()

    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Subclass should override this to compute features.

        Parameters:
            data (pd.DataFrame): MultiIndex [date, symbol] OHLCV data.

        Returns:
            pd.DataFrame: Indexed [date, symbol] with 1+ feature columns.
        """
        pass

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Entry point called by the pipeline.
        Logs timing, checks shape, and sanitizes output.
        """
        start = time.time()
        output = self.generate(data)

        if not isinstance(output, pd.DataFrame):
            raise ValueError(
                f"{self.__class__.__name__}.generate() must return a DataFrame"
            )

        if not isinstance(output.index, pd.MultiIndex):
            raise ValueError(
                f"{self.__class__.__name__} output must use MultiIndex ['date', 'symbol'], got {type(output.index)}"
            )

        if output.index.names != ["date", "symbol"]:
            raise ValueError(
                f"{self.__class__.__name__} output must have index ['date', 'symbol'], got {output.index.names}"
            )

        if output.isna().any().any():
            warnings.warn(
                f"{self.__class__.__name__}: contains NaNs, dropping them",
                stacklevel=2,
            )
            self.logger.warning(
                f"{self.__class__.__name__}: contains NaNs, dropping them"
            )
            output = output.dropna()

        duration = time.time() - start
        self.logger.debug(
            f"✅ {self.__class__.__name__} generated {output.shape[1]} feature(s) in {duration:.3f}s"
        )

        return output
