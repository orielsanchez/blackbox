import inspect
from typing import Type, TypeVar

from blackbox.config.loader import BacktestConfig
from blackbox.core.types.context import StrategyModels
from blackbox.models.interfaces import (
    AlphaModel,
    ExecutionModel,
    PortfolioConstructionModel,
    RiskModel,
    TransactionCostModel,
)
from blackbox.utils.registry import discover_models

T = TypeVar("T")


def build_models(config: BacktestConfig) -> StrategyModels:
    alpha_cls = _get_model_class(
        "blackbox.models.alpha", AlphaModel, config.alpha_model.name
    )
    risk_cls = _get_model_class(
        "blackbox.models.risk", RiskModel, config.risk_model.name
    )
    cost_cls = _get_model_class(
        "blackbox.models.cost", TransactionCostModel, config.tx_cost_model.name
    )
    portfolio_cls = _get_model_class(
        "blackbox.models.portfolio",
        PortfolioConstructionModel,
        config.portfolio_model.name,
    )
    execution_cls = _get_model_class(
        "blackbox.models.execution", ExecutionModel, config.execution_model.name
    )

    return StrategyModels(
        alpha=_safe_construct(alpha_cls, config.alpha_model.params),
        risk=_safe_construct(risk_cls, config.risk_model.params),
        cost=_safe_construct(cost_cls, config.tx_cost_model.params),
        portfolio=_safe_construct(portfolio_cls, config.portfolio_model.params),
        execution=_safe_construct(execution_cls, config.execution_model.params),
    )


def _get_model_class(package: str, interface: type, model_name: str) -> type:
    registry = discover_models(package, interface)
    key = model_name.lower()
    if key not in registry:
        raise ValueError(f"❌ Model '{model_name}' not found in {package}")
    cls = registry[key]
    if not issubclass(cls, interface):
        raise TypeError(
            f"❌ Model '{model_name}' in {package} does not subclass the interface"
        )
    return cls


def _safe_construct(cls: Type[T], params: dict) -> T:
    """Construct an instance of T, filtering out invalid kwargs."""
    if not params:
        return cls()

    signature = inspect.signature(cls.__init__)
    valid_params = {
        k: v for k, v in params.items() if k in signature.parameters and k != "self"
    }

    return cls(**valid_params)
