import pandas as pd

from blackbox.feature_generators.base import feature_registry
from blackbox.utils.context import get_logger


class FeaturePipeline:
    def __init__(self, features: list[dict]):
        """
        features: List of dicts like:
        [{"name": "momentum", "params": {"period": 5}}, {"name": "rolling_std", "params": {"period": 10}}]
        """
        self.generators = [
            feature_registry[f["name"]](**f.get("params", {})) for f in features
        ]
        self.logger = get_logger()

    def run(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Run all registered features on the full OHLCV window.
        Caller is responsible for slicing by current_date if needed.
        """
        feature_frames = []

        for generator in self.generators:
            name = generator.__class__.__name__
            try:
                output = generator.run(ohlcv)

                dates_in_output = output.index.get_level_values("date").unique()
                self.logger.info(
                    f"✅ {generator.__class__.__name__} returned {len(dates_in_output)} dates: {dates_in_output[:5].tolist()} ..."
                )

                if output.empty:
                    self.logger.warning(f"⚠️ {name}: no usable features returned")
                    continue

                self.logger.debug(f"✅ {name} output shape: {output.shape}")
                feature_frames.append(output)

            except KeyError as e:
                self.logger.warning(f"{name}: missing key during generation: {e}")
            except Exception as e:
                self.logger.error(f"❌ {name} failed: {e}", exc_info=True)

        if not feature_frames:
            self.logger.warning("⚠️ Feature pipeline produced no usable outputs")
            return pd.DataFrame()

        return pd.concat(feature_frames, axis=1)
