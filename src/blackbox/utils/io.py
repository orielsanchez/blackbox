import json
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from blackbox.core.types.context import BacktestMetrics
from blackbox.core.types.dataclasses import BacktestConfig
from blackbox.core.types.types import DailyLog


def write_results(
    logs: List[DailyLog],
    metrics: BacktestMetrics,
    config: BacktestConfig,
    output_dir: Path,
    equity_curve: pd.Series,
    plot_equity: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ────── Save daily logs ──────
    df_logs = pd.DataFrame(
        {
            "date": [log.date for log in logs],
            "equity": [log.equity for log in logs],
            "cash": [log.cash for log in logs],
            "pnl": [log.pnl for log in logs],
            "drawdown": [log.drawdown for log in logs],
            "num_trades": [log.trades.astype(bool).sum() for log in logs],
        }
    )
    df_logs.set_index("date", inplace=True)
    df_logs.sort_index(inplace=True)
    df_logs.to_csv(output_dir / "logs.csv")

    # ────── Save equity curve ──────
    if plot_equity and equity_curve is not None and not equity_curve.empty:
        equity_curve.to_csv(output_dir / "equity_curve.csv", header=["equity"])

        plt.figure(figsize=(10, 5))
        equity_curve.plot(title="Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.tight_layout()
        plt.savefig(output_dir / "equity_curve.png")
        plt.close()

    # ────── Save config.yaml ──────
    _dump_config_yaml(config, output_dir / "config.yaml")

    # ────── Save metrics + metadata ──────
    metadata = _extract_config_metadata(config)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"metrics": metrics.summary, "metadata": metadata}, f, indent=4)


def _dump_config_yaml(config: BacktestConfig, path: Path) -> None:
    """Dump the full config as a human-readable YAML file."""

    def _to_dict(obj: Any) -> Any:
        if isinstance(obj, list):
            return [_to_dict(i) for i in obj]
        elif hasattr(obj, "__dict__"):
            return {k: _to_dict(v) for k, v in obj.__dict__.items()}
        else:
            return obj

    with open(path, "w") as f:
        yaml.safe_dump(_to_dict(config), f, sort_keys=False)


def _extract_config_metadata(config: BacktestConfig) -> Dict[str, Any]:
    """Extract summary metadata from BacktestConfig for audit logging."""

    def safe_params(model_config: Any) -> Dict[str, Any]:
        return model_config.params if isinstance(model_config.params, dict) else {}

    return {
        "run_id": config.run_id,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "initial_portfolio_value": config.initial_portfolio_value,
        "risk_free_rate": config.risk_free_rate,
        "alpha_model": {
            "name": config.alpha_model.name,
            "params": safe_params(config.alpha_model),
        },
        "risk_model": {
            "name": config.risk_model.name,
            "params": safe_params(config.risk_model),
        },
        "tx_cost_model": {
            "name": config.tx_cost_model.name,
            "params": safe_params(config.tx_cost_model),
        },
        "portfolio_model": {
            "name": config.portfolio_model.name,
            "params": safe_params(config.portfolio_model),
        },
        "execution_model": {
            "name": config.execution_model.name,
            "params": safe_params(config.execution_model),
        },
    }


# ────── YAML Utility Functions ──────


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file and return a Python dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a Python dictionary to a YAML file."""
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
