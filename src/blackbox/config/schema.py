from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FeatureSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str
    params: Optional[Dict[str, Any]] = field(default_factory=dict)

    def get_feature_spec(self) -> List[FeatureSpec]:
        features = self.params.get("features", [])
        return [FeatureSpec(**f) for f in features]


@dataclass
class DataConfig:
    db_path: str
    rolling: bool = False
    window: int = 20


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

    # Runtime options
    min_holding_period: int = 0
    settlement_delay: int = 2
    initial_portfolio_value: float = 1_000_000
    plot_equity: bool = True
    risk_free_rate: float = 0.0

    # Logging
    log_level: str = "INFO"
    log_to_console: bool = True
    log_to_file: bool = True
    structured_logging: bool = False
