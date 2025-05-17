from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from blackbox.core.runtime.tracker import PositionMeta, PositionTracker
from blackbox.core.types.types import OHLCVSnapshot
from blackbox.models.interfaces import (
    AlphaModel,
    ExecutionModel,
    PortfolioConstructionModel,
    RiskModel,
    TransactionCostModel,
)
from blackbox.utils.logger import RichLogger

# ════════════════════════════════════════════════
# 🧩 STRATEGY CONFIGURATION
# ════════════════════════════════════════════════


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    params: dict[str, object] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.name, frozenset(self.params.items())))


@dataclass
class ModelConfig:
    name: str
    params: dict[str, object] = field(default_factory=dict)

    def get_feature_spec(self) -> list[FeatureSpec]:
        raw = self.params.get("features", [])
        return [FeatureSpec(**f) for f in raw] if isinstance(raw, list) else []

    def validate(self) -> None:
        if not self.name:
            raise ValueError("❌ ModelConfig is missing 'name'")
        if not isinstance(self.params, dict):
            raise TypeError("❌ ModelConfig 'params' must be a dictionary")


@dataclass
class DataConfig:
    db_path: str
    rolling: bool = False
    window: int = 20
    cache_path: Optional[str] = None
    force_reload: bool = False


@dataclass
class BacktestConfig:
    run_id: str
    start_date: str
    end_date: str
    universe_file: str
    data: DataConfig
    alpha_model: ModelConfig
    risk_model: ModelConfig
    tx_cost_model: ModelConfig
    portfolio_model: ModelConfig
    execution_model: ModelConfig

    # Runtime
    initial_portfolio_value: float = 1_000.0
    min_holding_period: int = 0
    settlement_delay: int = 2
    plot_equity: bool = True
    risk_free_rate: float = 0.0
    verbose: bool = False

    # Logging
    log_level: str = "INFO"
    log_to_console: bool = True
    log_to_file: bool = True
    structured_logging: bool = False


# ════════════════════════════════════════════════
# 💹 MARKET STATE & SNAPSHOTS
# ════════════════════════════════════════════════


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


# ════════════════════════════════════════════════
# 📈 TRACKER + LOGGING
# ════════════════════════════════════════════════


# ════════════════════════════════════════════════
# 📦 BACKTEST OUTPUT
# ════════════════════════════════════════════════


@dataclass
class BacktestMetrics:
    summary: dict[str, float]
    equity_curve: Optional[pd.Series] = None


# ════════════════════════════════════════════════
# 🧠 RUNTIME CONTEXT
# ════════════════════════════════════════════════


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
