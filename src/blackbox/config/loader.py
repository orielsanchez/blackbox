from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from dacite import from_dict

from blackbox.core.types.dataclasses import BacktestConfig


def load_config(path: Union[str, Path]) -> BacktestConfig:
    path = Path(path).resolve()

    with path.open("r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    config = from_dict(data_class=BacktestConfig, data=raw)

    # Validate all model configs immediately
    for model in [
        config.alpha_model,
        config.risk_model,
        config.tx_cost_model,
        config.portfolio_model,
        config.execution_model,
    ]:
        model.validate()

    return config


def dump_config(config: BacktestConfig, path: Union[str, Path]) -> None:
    path = Path(path).resolve()
    with path.open("w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
