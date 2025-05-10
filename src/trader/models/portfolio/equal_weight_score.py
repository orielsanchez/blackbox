import pandas as pd

from trader.core.portfolio import PortfolioConstructionModel
from trader.utils.schema import standardize_model_output


class EqualWeightScorePortfolioModel(PortfolioConstructionModel):
    def __init__(
        self,
        max_positions: int = 100,
        min_price: float = 2.0,
        min_volume: int = 1_000_000,
        max_notional_pct: float = 0.01,  # max 1% of capital per symbol
    ):
        self.max_positions = max_positions
        self.min_price = min_price
        self.min_volume = min_volume
        self.max_notional_pct = max_notional_pct

    def allocate(
        self,
        alpha_df: pd.DataFrame,
        risk_df: pd.DataFrame,
        tx_df: pd.DataFrame,
        slippage_df: pd.DataFrame,
        price_df: pd.DataFrame,
        capital: float,
    ) -> pd.DataFrame:
        """
        Selects top-N alpha symbols and assigns equal weight to each,
        subject to price, volume, and capital constraints.

        Inputs:
            alpha_df: ['symbol', 'alpha_score']
            price_df: ['symbol', 'price', 'volume']

        Output:
            DataFrame with ['symbol', 'weight', 'target_value', 'shares']
        """
        if "symbol" not in alpha_df or "alpha_score" not in alpha_df:
            raise ValueError("alpha_df must contain 'symbol' and 'alpha_score'")
        if "price" not in price_df.columns:
            raise ValueError("price_df must contain 'price'")

        df = alpha_df.copy()
        df["symbol"] = df["symbol"].astype(str)
        price_df["symbol"] = price_df["symbol"].astype(str)

        # Merge with prices and volumes
        df = df.merge(price_df[["symbol", "price"]], on="symbol", how="inner")
        if "volume" in price_df.columns:
            df = df.merge(price_df[["symbol", "volume"]], on="symbol", how="left")
            df["volume"] = df["volume"].fillna(0)
            df = df[df["volume"] >= self.min_volume]

        # Filter out low-price (e.g. penny) stocks
        df = df[df["price"] >= self.min_price]

        # Drop rows with missing values
        df = df.dropna(subset=["price", "alpha_score"])
        if df.empty:
            return pd.DataFrame(columns=["symbol", "weight", "target_value", "shares"])

        # Select top-N by alpha score
        df = df.sort_values("alpha_score", ascending=False).head(self.max_positions)
        if df.empty:
            return pd.DataFrame(columns=["symbol", "weight", "target_value", "shares"])

        # Equal weight allocation
        weight = 1.0 / len(df)
        df["weight"] = weight
        df["target_value"] = df["weight"] * capital

        # Apply max notional constraint
        max_notional = self.max_notional_pct * capital
        df["target_value"] = df["target_value"].clip(upper=max_notional)

        # Convert to share count
        df["shares"] = (df["target_value"] / df["price"]).fillna(0).astype(int)
        df["target_value"] = df["shares"] * df["price"]

        # Recalculate final weight
        total_allocated = df["target_value"].sum()
        if total_allocated > 0:
            df["weight"] = df["target_value"] / total_allocated
        else:
            df["weight"] = 0.0

        result = df[["symbol", "weight", "target_value", "shares"]].copy()
        result["weight"] = result["weight"].astype(float)
        result["target_value"] = result["target_value"].astype(float)
        result["shares"] = result["shares"].astype(int)
        return standardize_model_output(
            result,
            required_cols=["symbol", "weight", "target_value", "shares"],
            name="EqualWeightScorePortfolioModel",
        )

        return standardize_model_output(
            df[["symbol", "weight", "target_value", "shares"]],
            required_cols=["symbol", "weight", "target_value", "shares"],
            name="EqualWeightScorePortfolioModel",
        )
