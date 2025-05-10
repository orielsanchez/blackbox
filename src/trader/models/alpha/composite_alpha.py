import pandas as pd

from trader.core.alpha import AlphaModel
from trader.utils.schema import standardize_model_output


class CompositeAlphaModel(AlphaModel):
    def __init__(self, models: list[AlphaModel], weights: list[float] | None = None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        assert len(self.models) == len(self.weights), "Weights must match model count"

    def score(self, data: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        partials = []
        for model, weight in zip(self.models, self.weights):
            df = model.score(data, timestamp).copy()
            model_name = model.__class__.__name__
            df = df.rename(columns={"alpha_score": f"alpha_score_{model_name}"})
            df[f"weighted_score_{model_name}"] = (
                df[f"alpha_score_{model_name}"] * weight
            )
            partials.append(df)

        df = partials[0]
        for d in partials[1:]:
            df = df.merge(d, on="symbol", how="outer")

        df = df.fillna(0.0)
        weighted_cols = [col for col in df.columns if col.startswith("weighted_score_")]
        df["alpha_score"] = df[weighted_cols].sum(axis=1)

        return standardize_model_output(
            df, required_cols=["symbol", "alpha_score"], name="CompositeAlphaModel"
        ).merge(df.drop(columns="alpha_score"), on="symbol")
