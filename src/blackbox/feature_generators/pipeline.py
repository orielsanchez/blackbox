import pandas as pd

from blackbox.feature_generators.base import feature_registry


class FeaturePipeline:
    def __init__(self, features: list[dict]):
        """
        features: List of dicts like:
        [{"name": "momentum", "params": {"period": 5}}, {"name": "rolling_std", "params": {"period": 10}}]
        """
        self.generators = [
            feature_registry[f["name"]](**f.get("params", {})) for f in features
        ]

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs all registered features on input data.
        Returns a DataFrame with same index and all feature columns concatenated.
        """
        feature_frames = [gen.generate(data) for gen in self.generators]
        return pd.concat(feature_frames, axis=1)
