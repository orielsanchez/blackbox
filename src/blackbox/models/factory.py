from typing import Any, Type

from blackbox.config.schema import BacktestConfig
from blackbox.models.interfaces import (AlphaModel, ExecutionModel,
                                        PortfolioConstructionModel, RiskModel,
                                        TransactionCostModel)
from blackbox.models.registry_dynamic import discover_models

# Registry cache: prevent duplicate discovery
_discovered: dict[str, dict[str, type]] = {}


def _build_model(config_entry, model_dir: str, interface_cls: Type[Any]) -> Any:
    """Dynamically load and instantiate a model with given interface and params."""
    name = config_entry.name
    params = config_entry.params or {}

    # Cache registry
    if model_dir not in _discovered:
        _discovered[model_dir] = discover_models(model_dir, interface_cls)

    model_registry = _discovered[model_dir]

    if name not in model_registry:
        available = list(model_registry.keys())
        raise ValueError(
            f"❌ Model '{name}' not found in '{model_dir}'. Available: {available}"
        )

    model_cls = model_registry[name]
    try:
        return model_cls(**params)
    except Exception as e:
        raise RuntimeError(
            f"⚠️ Failed to instantiate model '{name}' in '{model_dir}': {e}"
        ) from e


def build_models(
    config: BacktestConfig,
) -> tuple[
    AlphaModel,
    RiskModel,
    TransactionCostModel,
    PortfolioConstructionModel,
    ExecutionModel,
]:
    alpha = _build_model(config.alpha_model, "src/blackbox/models/alpha", AlphaModel)
    risk = _build_model(config.risk_model, "src/blackbox/models/risk", RiskModel)
    cost = _build_model(
        config.tx_cost_model, "src/blackbox/models/cost", TransactionCostModel
    )
    portfolio = _build_model(
        config.portfolio_model,
        "src/blackbox/models/portfolio",
        PortfolioConstructionModel,
    )
    execution = _build_model(
        config.execution_model, "src/blackbox/models/execution", ExecutionModel
    )

    return alpha, risk, cost, portfolio, execution
