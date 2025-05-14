import pandas as pd

from blackbox.config.schema import FeatureSpec
from blackbox.feature_generators.base import feature_registry
from blackbox.feature_generators.pipeline import FeaturePipeline
from blackbox.utils.context import get_logger


class FeatureMatrixGenerator:
    def __init__(self, feature_spec: list[FeatureSpec]):
        self.generators = [feature_registry[f.name](**f.params) for f in feature_spec]

    def run(self, data: list[dict]) -> pd.DataFrame:
        feature_frames = []

        for snapshot in data:
            date = snapshot["date"]
            raw_ohlcv = snapshot["ohlcv"]

            # Ensure MultiIndex format
            if isinstance(raw_ohlcv, dict):
                df = pd.concat(
                    raw_ohlcv.values(), keys=raw_ohlcv.keys(), names=["symbol", "date"]
                )
                ohlcv = df.reset_index().set_index(["date", "symbol"]).sort_index()
            else:
                ohlcv = raw_ohlcv

            daily_features = [gen.generate(ohlcv) for gen in self.generators]

            if not daily_features:
                self.logger.warning(f"⚠️ No features generated for {date.date()}")
                continue

            day_df = pd.concat(daily_features, axis=1)

            # 🔧 Key Fix: Ensure full (date, symbol) MultiIndex
            day_df = day_df.reset_index()
            day_df["date"] = pd.Timestamp(date)  # Ensure uniform type
            day_df = day_df.set_index(["date", "symbol"]).sort_index()

            # Optional: Deduplication guard
            day_df = day_df[~day_df.index.duplicated(keep="first")]

            feature_frames.append(day_df)

        full_matrix = pd.concat(feature_frames).sort_index()

        # Final safety: drop dupes again just in case
        full_matrix = full_matrix[~full_matrix.index.duplicated(keep="first")]

        return full_matrix
