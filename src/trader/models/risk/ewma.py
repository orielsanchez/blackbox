import pandas as pd

from trader.core.risk import RiskModel


class EWMARiskModel(RiskModel):
    def score(self, data: pd.DataFrame) -> pd.DataFrame:
        vol = (
            data.groupby("symbol")["close"]
            .apple(lambda x: x.pct_change().ewm(span=20).std())
            .groupby("symbol")
            .last()
        )
        return pd.DataFrame({"risk_score": vol})
