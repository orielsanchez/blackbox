import pandas as pd

import blackbox.feature_generators
from blackbox.feature_generators.pipeline import FeaturePipeline
from blackbox.models.interfaces import AlphaModel


class MomentumAlphaModel(AlphaModel):
    name = "momentum_alpha"

    def __init__(self):
        self.feature_pipeline = FeaturePipeline(
            [
                {"name": "momentum", "params": {"period": 5}},
                {"name": "momentum", "params": {"period": 20}},
                {"name": "ema_crossover", "params": {"short": 10, "long": 50}},
            ]
        )

    def predict(self, snapshot: dict) -> pd.Series:
        """Alias for generate to support standard ML interface"""
        return self.generate(snapshot)

    def generate(self, snapshot: dict) -> pd.Series:
        """
        snapshot = {
            "date": pd.Timestamp,
            "prices": pd.Series,
            "ohlcv": pd.DataFrame  # MultiIndex: (date, symbol)
        }
        Returns:
            pd.Series [symbol → signal]
        """
        ohlcv = snapshot["ohlcv"]
        today = snapshot["date"]

        # Run pipeline
        features = self.feature_pipeline.run(ohlcv)

        # Get latest feature values
        try:
            f = features.loc[today]
        except KeyError:
            return pd.Series(dtype=float)

        # Combine signals: weighted average
        score = (
            0.4 * f["momentum_5d"] + 0.4 * f["momentum_20d"] + 0.2 * f["ema_10_50_diff"]
        )

        # Optional: Zero out weak signals
        score = score.where(score.abs() > 0.01, 0)

        return score.fillna(0)
