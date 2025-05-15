import pandas as pd

from blackbox.feature_generators.base import feature_registry
from blackbox.feature_generators.utils import validate_feature_output
from blackbox.utils.context import get_logger


class FeaturePipeline:
    def __init__(self, features: list[dict]):
        """
        features: List of dicts like:
        [{"name": "momentum", "params": {"period": 5}}, {"name": "rolling_std", "params": {"period": 10}}]
        """
        self.generators = [feature_registry[f["name"]](**f.get("params", {})) for f in features]
        self.logger = get_logger()

    def run(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Run all registered features on the full OHLCV window.
        Caller is responsible for slicing by current_date if needed.
        """
        feature_frames = []

        for feature in self.generators:
            name = feature.__class__.__name__
            try:
                raw_output = feature.generate(ohlcv)

                if raw_output is None or raw_output.empty:
                    self.logger.warning(f"⚠️ {name}: empty output")
                    continue

                self.logger.debug(f"{name} raw output tail:\n{raw_output.tail()}")

                validated_output = validate_feature_output(
                    name,
                    raw_output,
                    current_date=None,  # Let validator handle indexing
                )

                if validated_output.empty:
                    self.logger.warning(f"⚠️ {name}: no usable features after validation")
                    continue

                self.logger.debug(f"✅ {name} output shape: {validated_output.shape}")
                feature_frames.append(validated_output)

            except KeyError as e:
                self.logger.warning(f"{name}: missing key during generation: {e}")
            except Exception as e:
                self.logger.error(f"❌ {name} failed: {e}")

        if not feature_frames:
            self.logger.warning("⚠️ Feature pipeline produced no usable outputs")
            return pd.DataFrame()

        return pd.concat(feature_frames, axis=1)
