from dataclasses import dataclass
from typing import Optional

import pandas as pd

from blackbox.core.runtime.tracker import PositionMeta, PositionTracker
from blackbox.core.types.dataclasses import BacktestConfig
from blackbox.core.types.types import OHLCVSnapshot
from blackbox.models.interfaces import (
    AlphaModel,
    ExecutionModel,
    PortfolioConstructionModel,
    RiskModel,
    TransactionCostModel,
)
from blackbox.utils.logger import RichLogger


@dataclass
class FeatureMatrixInfo:
    features: list[str]
    dates: set[pd.Timestamp]
    min_date: pd.Timestamp
    warmup: int


@dataclass
class PreparedDataBundle:
    snapshots: list[OHLCVSnapshot]
    feature_matrix: pd.DataFrame
    metadata: FeatureMatrixInfo


@dataclass
class ExecutionState:
    equity: float
    cash: float
    positions: dict[str, PositionMeta]


@dataclass
class BacktestMetrics:
    summary: dict[str, float]
    equity_curve: Optional[pd.Series] = None


@dataclass
class StrategyModels:
    alpha: AlphaModel
    risk: RiskModel
    cost: TransactionCostModel
    portfolio: PortfolioConstructionModel
    execution: ExecutionModel


@dataclass
class StrategyContext:
    config: BacktestConfig
    logger: RichLogger
    models: StrategyModels
    tracker: PositionTracker
    initial_equity: float
    current_date: Optional[pd.Timestamp] = None
