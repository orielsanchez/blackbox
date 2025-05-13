from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    name: str
    params: Optional[Dict] = None


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
    min_holding_period: int = 0
    settlement_delay: int = 2

    # Logging
    log_level: str = "INFO"
    log_to_console: bool = True
    log_to_file: bool = True
    structured_logging: bool = False
