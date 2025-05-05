import pandas as pd

from trader.core.risk import RiskModel
from trader.utils.schema import standardize_model_output


class EWMARiskModel(RiskModel):
    def __init__(self, span: int = 20):
        self.span = span

    def score(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        """
        Compute EWMA volatility (risk score) for each symbol up to the given timestamp.

        Parameters:
            data: DataFrame with columns ['symbol', 'timestamp', 'price']
            timestamp: compute risk scores using data <= this timestamp

        Returns:
            DataFrame with columns ['symbol', 'risk_score']
        """
        required_cols = {"symbol", "timestamp", "price"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Missing columns: {required_cols - set(data.columns)}")

        data = data.copy()
        data = data[data["timestamp"] <= timestamp]
        data = data.dropna(subset=["price"])
        data = data.sort_values(["symbol", "timestamp"])

        risk_scores = {}

        for symbol, group in data.groupby("symbol"):
            if len(group) < self.span + 1:
                risk_scores[symbol] = 0.0
                continue

            returns = group["price"].pct_change()
            ewma_vol = returns.ewm(span=self.span).std().iloc[-1]

            risk_scores[symbol] = ewma_vol if pd.notnull(ewma_vol) else 0.0

        df = pd.DataFrame.from_dict(
            risk_scores, orient="index", columns=["risk_score"]
        ).reset_index(names="symbol")

        return standardize_model_output(
            df, required_cols=["symbol", "risk_score"], name="EWMARiskModel"
        )
