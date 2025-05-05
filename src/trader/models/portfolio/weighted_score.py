import pandas as pd

from trader.core.portfolio import PortfolioConstructionModel
from trader.utils.schema import standardize_model_output


class WeightedScorePortfolioModel(PortfolioConstructionModel):
    def __init__(
        self,
        max_weight: float = 0.1,
        min_dollar_value: float = 500.0,
        min_positions: int = 3,
    ):
        self.max_weight = max_weight
        self.min_dollar_value = min_dollar_value
        self.min_positions = min_positions

    def allocate(
        self,
        alpha_df: pd.DataFrame,
        risk_df: pd.DataFrame,
        tx_df: pd.DataFrame,
        slippage_df: pd.DataFrame,
        price_df: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        for name, df in {
            "alpha": alpha_df,
            "risk": risk_df,
            "tx": tx_df,
            "slippage": slippage_df,
            "price": price_df,
        }.items():
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].astype(str)
            else:
                raise ValueError(f"{name} DataFrame missing 'symbol' column")

        df = (
            alpha_df.merge(risk_df, on="symbol")
            .merge(tx_df, on="symbol")
            .merge(slippage_df, on="symbol")
            .merge(price_df, on="symbol")
        )

        # === Score adjustment ===
        df["adjusted"] = df["alpha_score"] / (df["risk_score"] + 1e-6)
        df["adjusted"] -= (df["tx_cost"] + df["slippage"]) / capital
        df = df[df["adjusted"] > 0]

        if df.empty:
            return pd.DataFrame(columns=["symbol", "weight", "target_value", "shares"])

        # === Initial weight allocation ===
        df["weight"] = df["adjusted"] / df["adjusted"].sum()
        df["weight"] = df["weight"].clip(upper=self.max_weight)
        df["weight"] /= df["weight"].sum()

        df["target_value"] = df["weight"] * capital
        df["shares"] = (df["target_value"] / df["price"]).astype(int)
        df["actual_value"] = df["shares"] * df["price"]

        # === Filter out positions below minimum size ===
        df = df[(df["shares"] > 0) & (df["actual_value"] >= self.min_dollar_value)]

        # If we don't have enough positions, fallback to top alpha picks
        if len(df) < self.min_positions:
            fallback = (
                alpha_df.merge(price_df, on="symbol")
                .sort_values("alpha_score", ascending=False)
                .copy()
            )
            fallback["target_value"] = self.max_weight * capital
            fallback["shares"] = (fallback["target_value"] / fallback["price"]).astype(
                int
            )
            fallback = fallback[fallback["shares"] > 0]

            # Select at least one fallback even if still below min dollar value
            if fallback.empty:
                print("⚠️  No fallback trades possible — prices too high?")
            else:
                fallback = fallback.head(self.min_positions)
                fallback["weight"] = fallback["shares"] * fallback["price"] / capital
                fallback["target_value"] = fallback["shares"] * fallback["price"]
                df = fallback[["symbol", "weight", "target_value", "shares"]]

        return standardize_model_output(
            df[["symbol", "weight", "target_value", "shares"]],
            required_cols=["symbol", "weight", "target_value", "shares"],
            name="WeightedScorePortfolioModel",
        )
