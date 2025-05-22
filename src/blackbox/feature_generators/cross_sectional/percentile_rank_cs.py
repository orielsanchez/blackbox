import pandas as pd
from scipy.stats import rankdata

from blackbox.feature_generators.base import BaseFeatureGenerator, register_feature


@register_feature("percentile_rank_cs")
class PercentileRankCSFeature(BaseFeatureGenerator):
    def __init__(self, column: str = "close"):
        """
        Computes the percentile rank of each symbol's value within the cross-section (same date).

        Args:
            column: The column to rank across symbols (e.g. "close", "volume").
        """
        super().__init__()
        self.column = column

    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column not in data.columns:
            raise ValueError(f"Input data must include '{self.column}' column.")
        if "symbol" not in data.index.names or "date" not in data.index.names:
            raise ValueError("Input data must have MultiIndex with levels: ['date', 'symbol']")

        values = data[self.column]

        def compute_percentile_rank(group: pd.Series) -> pd.Series:
            ranks = rankdata(group, method="average")  # 1-based
            percentile = (ranks - 1) / (len(group) - 1) if len(group) > 1 else 0.0
            return pd.Series(percentile, index=group.index)

        ranked = values.groupby(level="date").transform(compute_percentile_rank)
        ranked.name = f"{self.column}_percentile_rank_cs"
        ranked.index = data.index

        return ranked.to_frame()
