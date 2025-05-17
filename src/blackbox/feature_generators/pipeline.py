from typing import Any

import pandas as pd

from blackbox.feature_generators.base import feature_registry
from blackbox.utils.context import get_logger


class FeaturePipeline:
    def __init__(self, features: list[dict[str, Any]]):
        """
        Initialize a pipeline of feature generators.

        Parameters:
            features: A list of dicts like:
              [
                {"name": "momentum", "params": {"period": 5}},
                {"name": "rolling_std", "params": {"period": 10}},
              ]
        """
        self.logger = get_logger()
        self.generators = [
            feature_registry[f["name"]](**f.get("params", {})) for f in features
        ]

        if not self.generators:
            self.logger.warning("⚠️ No feature generators registered")

    def run(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Run all features on the provided OHLCV data.

        Parameters:
            ohlcv: A MultiIndex DataFrame with index [date, symbol] and columns ['open', 'high', 'low', 'close', 'volume']

        Returns:
            A MultiIndex DataFrame [date, symbol] with concatenated feature columns.
        """
        if not self.generators:
            return pd.DataFrame()

        feature_frames = []

        for generator in self.generators:
            name = generator.__class__.__name__
            try:
                output = generator.generate(ohlcv)  # 🔁 FIXED: .generate(), not .run()

                if output.empty:
                    self.logger.warning(f"⚠️ {name}: no usable features returned")
                    continue

                dates_in_output = output.index.get_level_values("date").unique()
                self.logger.info(
                    f"✅ {name} output: {len(dates_in_output)} dates | shape: {output.shape}"
                )

                feature_frames.append(output)

            except KeyError as e:
                self.logger.warning(f"⚠️ {name} missing key: {e}")
            except Exception as e:
                self.logger.error(f"❌ {name} failed: {e}", exc_info=True)

        if not feature_frames:
            self.logger.warning("⚠️ Feature pipeline produced no usable outputs")
            return pd.DataFrame()

        return pd.concat(feature_frames, axis=1)
