import inspect
from typing import Any, Dict, Type, TypeVar

from blackbox.core.types.context import StrategyModels
from blackbox.core.types.dataclasses import BacktestConfig, ModelConfig
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
    return StrategyModels(
        alpha=_build_model("blackbox.models.alpha", AlphaModel, config.alpha_model),
        risk=_build_model("blackbox.models.risk", RiskModel, config.risk_model),
        cost=_build_model("blackbox.models.cost", TransactionCostModel, config.tx_cost_model),
        portfolio=_build_model(
            "blackbox.models.portfolio",
            PortfolioConstructionModel,
            config.portfolio_model,
        ),
        execution=_build_model("blackbox.models.execution", ExecutionModel, config.execution_model),
    )


def _build_model(package: str, interface: Type, model_config: ModelConfig) -> Any:
    cls = _get_model_class(package, interface, model_config.name)
    return _safe_construct(cls, model_config.params)


def _get_model_class(package: str, interface: Type, model_name: str) -> Type:
    registry = discover_models(package, interface)
    key = model_name.lower()
    if key not in registry:
        raise ValueError(f"❌ Model '{model_name}' not found in {package}")
    cls = registry[key]
    if not issubclass(cls, interface):
        raise TypeError(
            f"❌ Model '{model_name}' in {package} does not subclass {interface.__name__}"
        )
    return cls


def _safe_construct(cls: Type[T], params: Dict[str, Any]) -> T:
    """Construct an instance of T, recursively instantiating any submodel config dicts."""
    if not params:
        return cls()

    signature = inspect.signature(cls.__init__)
    valid_keys = set(signature.parameters) - {"self"}

    resolved_params = {}
    for key, value in params.items():
        if key not in valid_keys:
            continue

        # Handle nested submodels (dict with 'name' and 'params')
        if isinstance(value, dict) and "name" in value and "params" in value:
            sub_cls = _get_model_class(
                _infer_model_package(key), _infer_model_interface(key), value["name"]
            )
            resolved_params[key] = _safe_construct(sub_cls, value["params"])
        else:
            resolved_params[key] = value

    return cls(**resolved_params)


def _infer_model_package(param_key: str) -> str:
    if "alpha" in param_key or "momentum" in param_key or "mean" in param_key:
        return "blackbox.models.alpha"
    if "risk" in param_key:
        return "blackbox.models.risk"
    if "cost" in param_key or "tx" in param_key:
        return "blackbox.models.cost"
    if "portfolio" in param_key:
        return "blackbox.models.portfolio"
    if "execution" in param_key:
        return "blackbox.models.execution"
    raise ValueError(f"❌ Cannot infer model package from param key: '{param_key}'")


def _infer_model_interface(param_key: str) -> Type:
    if "alpha" in param_key or "momentum" in param_key or "mean" in param_key:
        return AlphaModel
    if "risk" in param_key:
        return RiskModel
    if "cost" in param_key or "tx" in param_key:
        return TransactionCostModel
    if "portfolio" in param_key:
        return PortfolioConstructionModel
    if "execution" in param_key:
        return ExecutionModel
    raise ValueError(f"❌ Cannot infer model interface from param key: '{param_key}'")
