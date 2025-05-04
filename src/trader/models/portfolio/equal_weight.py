import pandas as pd

from trader.core.portfolio import PortfolioConstructionModel


class EqualWeightPortfolioModel(PortfolioConstructionModel):
    def __init__(self, top_n: int = 10):
        self.top_n = top_n

    def allocate(
        self,
        alpha_scores: pd.DataFrame,
        risk_scores: pd.DataFrame,
        cost_scores: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        df = alpha_scores.copy()
        df = df.sort_values("alpha_score", ascending=False).head(self.top_n)
        df["target_weight"] = 1.0 / len(df)
        return df[["target_weight"]]
