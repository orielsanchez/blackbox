from typing import Any, List

from blackbox.core.types.dataclasses import BacktestConfig, FeatureSpec, ModelConfig


def extract_features_from_model(model: Any) -> List[FeatureSpec]:
    specs: List[FeatureSpec] = []

    if isinstance(model, ModelConfig):
        params = model.params or {}

        # If this model has top-level features
        if "features" in params and isinstance(params["features"], list):
            for f in params["features"]:
                if isinstance(f, dict):
                    specs.append(FeatureSpec(name=f["name"], params=f.get("params", {})))
                elif isinstance(f, FeatureSpec):
                    specs.append(f)

        # Look inside possible submodels recursively
        for val in params.values():
            if isinstance(val, dict) and "name" in val and "params" in val:
                submodel = ModelConfig(name=val["name"], params=val["params"])
                specs.extend(extract_features_from_model(submodel))

    return specs


def collect_all_feature_specs(config: BacktestConfig) -> List[FeatureSpec]:
    """
    Collects all unique feature specs from all models in the backtest config.
    """
    feature_specs = []

    for model_config in [
        config.alpha_model,
        config.risk_model,
        config.tx_cost_model,
        config.portfolio_model,
        config.execution_model,
    ]:
        feature_specs.extend(extract_features_from_model(model_config))

    # Deduplicate by FeatureSpec hash
    return list(set(feature_specs))
