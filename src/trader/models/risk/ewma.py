import pandas as pd

from trader.core.risk import RiskModel
from trader.utils.schema import standardize_model_output


class EWMARiskModel(RiskModel):
    def __init__(self, span: int = 20, min_span: int = 3, strict: bool = False):
        """
        span: ideal span to use for EWMA volatility
        min_span: minimum span allowed if adapting to short histories
        strict: if True, use only exact span and fallback to 0.0 if not enough data
        """
        self.span = span
        self.min_span = min_span
        self.strict = strict

    def score(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        required_cols = {"symbol", "timestamp", "price"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"Missing columns: {required_cols - set(data.columns)}")

        data = data.copy()
        data = data[data["timestamp"] <= timestamp]
        data = data.dropna(subset=["price"])
        data = data.sort_values(["symbol", "timestamp"])

        risk_scores = {}

        for symbol, group in data.groupby("symbol"):
            n = len(group)

            # Choose span adaptively if not strict
            if self.strict:
                if n < self.span + 1:
                    risk_scores[symbol] = 0.0
                    continue
                used_span = self.span
            else:
                used_span = max(min(self.span, n - 1), self.min_span)
                if n < used_span + 1:
                    risk_scores[symbol] = 0.0
                    continue

            returns = group["price"].pct_change()
            ewma_vol = returns.ewm(span=used_span).std().iloc[-1]

            if pd.isna(ewma_vol) or ewma_vol == 0.0:
                risk_scores[symbol] = 0.0
            else:
                risk_scores[symbol] = ewma_vol

        df = pd.DataFrame.from_dict(
            risk_scores, orient="index", columns=["risk_score"]
        ).reset_index(names="symbol")

        if df["risk_score"].abs().sum() == 0.0:
            print("⚠️ All risk scores are 0. Check data span or price stability.")

        return standardize_model_output(
            df, required_cols=["symbol", "risk_score"], name="EWMARiskModel"
        )
