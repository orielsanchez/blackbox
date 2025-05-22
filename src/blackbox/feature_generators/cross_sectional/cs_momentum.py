import pandas as pd
from scipy.stats import rankdata

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("cs_momentum")
class CrossSectionalMomentumFeature(BaseFeatureGenerator):
    def __init__(self, period: int = 5, normalize: bool = True):
        """
        Computes cross-sectional momentum by ranking N-day returns across symbols.

        Args:
            period: Lookback period for return (in trading days).
            normalize: If True, output is scaled to [0.0, 1.0] using percentile rank.
                       If False, raw cross-sectional ranks are returned.
        """
        super().__init__()
        self.period = period
        self.normalize = normalize

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if "close" not in data.columns:
            raise ValueError("Input data must include 'close' column.")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError("Input data must have MultiIndex with levels: ['date', 'symbol']")

        close = data["close"]
        grouped = close.groupby(level="symbol")

        # Compute past return
        past_return = grouped.transform(lambda x: x / x.shift(self.period) - 1.0)

        def rank_cs(group: pd.Series) -> pd.Series:
            ranks = rankdata(group, method="average")
            if self.normalize:
                return (
                    pd.Series((ranks - 1) / (len(group) - 1), index=group.index)
                    if len(group) > 1
                    else pd.Series(0.0, index=group.index)
                )
            return pd.Series(ranks, index=group.index)

        ranked_momentum = past_return.groupby(level="date").transform(rank_cs)
        name = f"cs_momentum_{self.period}{'_pct' if self.normalize else '_rank'}"
        ranked_momentum.name = name
        ranked_momentum.index = data.index

        return ranked_momentum.to_frame()
