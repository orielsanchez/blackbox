from pathlib import Path

import yaml
from dacite import from_dict

from blackbox.config.schema import BacktestConfig


def load_config(path: str | Path) -> BacktestConfig:
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return from_dict(data_class=BacktestConfig, data=raw)
