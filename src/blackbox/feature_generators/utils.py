from typing import List

from blackbox.core.types.context import BacktestConfig, FeatureSpec


def collect_all_feature_specs(config: BacktestConfig) -> List[FeatureSpec]:
    """
    Collects all unique feature specs from all models in the backtest config.

    Args:
        config: The backtest config

    Returns:
        A list of unique FeatureSpec instances
    """
    feature_specs = []

    def add_specs(model_config):
        features = model_config.get_feature_spec()
        feature_specs.extend(features)

    add_specs(config.alpha_model)
    add_specs(config.risk_model)
    add_specs(config.tx_cost_model)
    add_specs(config.portfolio_model)
    add_specs(config.execution_model)

    # Deduplicate by feature name + params (using FeatureSpec.__hash__)
    unique_specs = list(set(feature_specs))
    return unique_specs
