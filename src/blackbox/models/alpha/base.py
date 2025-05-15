from abc import ABC

import pandas as pd

from blackbox.feature_generators.resolve import (resolve_and_generate_features,
                                                 resolve_feature_columns)
from blackbox.models.interfaces import AlphaModel
from blackbox.utils.context import get_feature_matrix, get_logger


class FeatureAwareAlphaModel(AlphaModel, ABC):
    def __init__(self, features: list[dict]):
        self.logger = get_logger()
        self.feature_config = features or []
        self.feature_columns = resolve_feature_columns(self.feature_config)

    def get_feature_matrix_for(self, snapshot: dict) -> pd.DataFrame:
        feature_matrix = get_feature_matrix()
        ohlcv = snapshot["ohlcv"]
        return resolve_and_generate_features(self.feature_config, feature_matrix, ohlcv)
