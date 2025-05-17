from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd

from blackbox.core.types.types import OHLCVSnapshot, PortfolioTarget, TradeResult


class AlphaModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model name."""
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Predict raw alpha signals given today's features.

        Args:
            features: Feature matrix for the current day, indexed by (date, symbol).

        Returns:
            Raw signal vector indexed by (date, symbol).
        """
        pass


class RiskModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model name."""
        pass

    @abstractmethod
    def apply(self, signals: pd.Series, features: pd.DataFrame) -> pd.Series:
        """
        Apply risk constraints to raw alpha signals.

        Args:
            signals: Raw alpha signal vector.
            features: Feature matrix for the current day or lookback window.

        Returns:
            Risk-adjusted signal vector.
        """
        pass


class TransactionCostModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model name."""
        pass

    @abstractmethod
    def adjust(self, signals: pd.Series, features: pd.DataFrame) -> pd.Series:
        """
        Adjust signals to account for transaction costs.

        Args:
            signals: Portfolio target weights (before cost adjustments).
            features: Feature matrix for the current day.

        Returns:
            Cost-adjusted portfolio weights.
        """
        pass


class PortfolioConstructionModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model name."""
        pass

    @abstractmethod
    def construct(
        self,
        signals: pd.Series,
        capital: float,
        features: pd.DataFrame,
        snapshot: OHLCVSnapshot,
    ) -> PortfolioTarget:
        """
        Generate final portfolio target weights.

        Args:
            signals: Cost-adjusted signals.
            capital: Total capital available.
            features: Feature matrix for the current day or lookback window.
            snapshots: OHLCVSnapshot.

        Returns:
            PortfolioTarget with weights and metadata.
        """
        pass

    @abstractmethod
    def feedback_from_execution(self, feedback: Dict[str, Dict[str, object]]) -> None:
        """
        Optionally handle execution feedback from the simulation/live system.

        Args:
            feedback: Execution result feedback keyed by symbol.
        """
        pass


class ExecutionModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model name."""
        pass

    @property
    @abstractmethod
    def portfolio_value(self) -> float:
        """Latest known portfolio equity value."""
        pass

    @property
    @abstractmethod
    def current_cash(self) -> float:
        """Latest known cash balance."""
        pass

    @abstractmethod
    def execute(
        self,
        target: PortfolioTarget,
        current: pd.Series,
        snapshot: OHLCVSnapshot,
        prices: pd.Series,
        open_prices: pd.Series,
        date: pd.Timestamp,
    ) -> TradeResult:
        """
        Execute trades to move from current to target weights.

        Args:
            target: Desired portfolio weights.
            current: Current portfolio weights.
            snapshot: OHLCVSnapshot.
            prices: End-of-day prices.
            open_prices: Next-day open prices (for slippage).
            date: Simulation date.

        Returns:
            TradeResult containing fills and metadata.
        """
        pass

    @abstractmethod
    def record(self, trades: pd.Series, feedback: Dict) -> None:
        """
        Store trade execution results and metadata.

        Args:
            trades: Executed trades.
            feedback: Additional feedback for logging or learning.
        """
        pass

    @abstractmethod
    def update_portfolio(
        self,
        current: pd.Series,
        trades: pd.Series,
        capital: float,
        prices: pd.Series,
    ) -> pd.Series:
        """
        Update portfolio state given executed trades and prices.

        Args:
            current: Portfolio before trades.
            trades: Executed trades.
            capital: Available capital.
            prices: End-of-day prices.

        Returns:
            New portfolio weights.
        """
        pass

    @abstractmethod
    def mark_to_market(self, prices: pd.Series) -> None:
        """
        Update internal valuation based on current prices.

        Args:
            prices: Latest market prices.
        """
        pass

    @abstractmethod
    def get_available_capital(self) -> float:
        """
        Return capital currently available for trading.

        Returns:
            Available cash or buying power.
        """
        pass
