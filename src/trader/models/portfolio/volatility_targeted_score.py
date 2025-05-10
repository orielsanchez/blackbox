import pandas as pd

from trader.core.portfolio import PortfolioConstructionModel


class VolatilityTargetedScorePortfolioModel(PortfolioConstructionModel):
    def __init__(
        self,
        weights: dict,
        min_alpha: float = 0.0,
        top_n: int | None = None,
        volatility_column: str = "risk_score",
        volatility_targeting: bool = True,
        volatility_floor: float = 1e-4,  # prevents divide-by-zero
    ):
        self.weights = weights
        self.min_alpha = min_alpha
        self.top_n = top_n
        self.volatility_column = volatility_column
        self.volatility_targeting = volatility_targeting
        self.volatility_floor = volatility_floor

    def allocate(
        self,
        alpha_df: pd.DataFrame,
        risk_df: pd.DataFrame,
        tx_df: pd.DataFrame,
        slippage_df: pd.DataFrame,
        price_df: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        df = alpha_df.copy()
        for d in [risk_df, tx_df, slippage_df, price_df]:
            df = df.merge(d, on="symbol", how="outer")

        df = df.dropna()
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "weight",
                    "target_value",
                    "price",
                    "shares",
                    "combined_score",
                ]
            )

        # Optional alpha filter
        if "alpha_score" in df.columns:
            df = df[df["alpha_score"] > self.min_alpha]

        # Normalize each score
        for col in self.weights:
            col_min = df[col].min()
            col_max = df[col].max()
            df[f"{col}_norm"] = (
                (df[col] - col_min) / (col_max - col_min) if col_max > col_min else 0.0
            )

        # Weighted sum of scores
        df["combined_score"] = 0.0
        for col, weight in self.weights.items():
            norm_col = f"{col}_norm"
            df[f"{col}_contrib"] = weight * df[norm_col]
            df["combined_score"] += df[f"{col}_contrib"]

        df = df[df["combined_score"] > 0]
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "weight",
                    "target_value",
                    "price",
                    "shares",
                    "combined_score",
                ]
            )

        # Optional: top-N filtering
        if self.top_n:
            df = df.sort_values("combined_score", ascending=False).head(self.top_n)

        # Volatility targeting (scale score-based weights by inverse volatility)
        if self.volatility_targeting and self.volatility_column in df.columns:
            df["inv_vol"] = 1 / (df[self.volatility_column] + self.volatility_floor)
            df["adjusted_score"] = df["combined_score"] * df["inv_vol"]
        else:
            df["adjusted_score"] = df["combined_score"]

        # Final weights and allocations
        df["weight"] = df["adjusted_score"] / df["adjusted_score"].sum()
        df["target_value"] = df["weight"] * capital
        df["shares"] = (df["target_value"] / df["price"]).astype(int)

        return df[
            ["symbol", "weight", "target_value", "price", "shares", "combined_score"]
            + [f"{col}_norm" for col in self.weights]
            + [f"{col}_contrib" for col in self.weights]
            + (["inv_vol"] if self.volatility_targeting else [])
        ]
