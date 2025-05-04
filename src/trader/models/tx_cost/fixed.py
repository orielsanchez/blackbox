import pandas as pd

from trader.core.tx_cost import TransactionCostModel


class FixedCostModel(TransactionCostModel):
    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"tx_cost_score": 0.001}, index=data["symbol"].unique())
