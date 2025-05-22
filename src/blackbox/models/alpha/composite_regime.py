from typing import Dict, List, Optional

import pandas as pd

from blackbox.models.alpha.mean_reversion import MeanReversionAlphaModel
from blackbox.models.alpha.momentum import MomentumAlphaModel  # To be created
from blackbox.models.interfaces import AlphaModel
from blackbox.utils.context import get_logger
from blackbox.utils.signals import normalize_signal


class CompositeRegimeAlphaModel(AlphaModel):
    def __init__(
        self,
        mean_reversion_model: Optional[AlphaModel] = None,
        momentum_model: Optional[AlphaModel] = None,
        regime_feature: str = "adx_14",
        trend_threshold: float = 25.0,
        universe: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.mean_reversion_model = mean_reversion_model or MeanReversionAlphaModel(
            universe=universe, verbose=verbose
        )
        self.momentum_model = momentum_model or MomentumAlphaModel(
            universe=universe, verbose=verbose
        )
        self.regime_feature = regime_feature
        self.trend_threshold = trend_threshold
        self.universe = universe or []
        self.verbose = verbose
        self.logger = get_logger()

    @property
    def name(self) -> str:
        return "composite_regime_aware_alpha"

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not isinstance(features.index, pd.MultiIndex):
            raise ValueError("Expected MultiIndex [date, symbol]")

        date = features.index.get_level_values("date").unique()
        if len(date) != 1:
            raise ValueError(f"Expected single date, got {date.tolist()}")
        date = date[0]

        symbols = (
            features.index.get_level_values("symbol")
            .unique()
            .intersection(self.universe)
            if self.universe
            else features.index.get_level_values("symbol").unique()
        )

        idx = pd.MultiIndex.from_product([[date], symbols], names=["date", "symbol"])

        if features.empty or self.regime_feature not in features.columns:
            self.logger.warning(
                f"{date} | ⚠️ Missing regime feature: {self.regime_feature}"
            )
            return pd.Series(0.0, index=idx, name="composite_signal")

        features_today = features.loc[pd.IndexSlice[date, symbols], :]

        regime_values = features_today[self.regime_feature]
        trending = regime_values >= self.trend_threshold
        mean_reverting = regime_values < self.trend_threshold

        # Dispatch models by regime
        trending_symbols = (
            regime_values[trending].index.get_level_values("symbol").unique()
        )
        mean_rev_symbols = (
            regime_values[mean_reverting].index.get_level_values("symbol").unique()
        )

        trend_signals = pd.Series(0.0, index=idx, name="composite_signal")
        mean_rev_signals = pd.Series(0.0, index=idx, name="composite_signal")

        if not trending_symbols.empty:
            trend_slice = features.loc[pd.IndexSlice[date, trending_symbols], :]
            trend_signals = self.momentum_model.predict(trend_slice)

        if not mean_rev_symbols.empty:
            mean_rev_slice = features.loc[pd.IndexSlice[date, mean_rev_symbols], :]
            mean_rev_signals = self.mean_reversion_model.predict(mean_rev_slice)

        # Combine both sets of signals
        combined = pd.Series(0.0, index=idx, name="composite_signal")
        combined.update(trend_signals)
        combined.update(mean_rev_signals)

        return normalize_signal(combined)
