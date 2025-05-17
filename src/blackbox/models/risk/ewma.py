import pandas as pd

from blackbox.models.interfaces import RiskModel


class EWMAVolatilityRisk(RiskModel):
    def __init__(
        self,
        volatility_column: str = "ewm_vol_20",
        max_position_size: float = 0.25,
        max_leverage: float = 1.0,
        allow_shorts: bool = True,
        epsilon: float = 1e-6,
    ):
        """
        Risk model that penalizes high-volatility names using inverse EWMA volatility.

        Args:
            volatility_column: Column in features to use (must be present).
            max_position_size: Cap per symbol.
            max_leverage: Total leverage cap.
            allow_shorts: If False, signals are clipped to [0, max_position_size].
            epsilon: Small float to prevent divide-by-zero.
        """
        self._vol_col = volatility_column
        self._max_pos = max_position_size
        self._max_lev = max_leverage
        self._allow_shorts = allow_shorts
        self._epsilon = epsilon

    @property
    def name(self) -> str:
        return "ewma_volatility"

    def apply(self, signals: pd.Series, features: pd.DataFrame) -> pd.Series:
        if signals.empty:
            return pd.Series(dtype=float)

        # Extract current date
        dates = signals.index.get_level_values("date").unique()
        if len(dates) != 1:
            raise ValueError(f"RiskModel expects one date per call, got: {dates}")
        date = dates[0]

        # Flatten
        day_signals = signals.loc[date]
        day_features = features.loc[date]

        if self._vol_col not in day_features.columns:
            raise ValueError(f"Missing column '{self._vol_col}' in features")

        # Extract volatility series
        volatility = day_features[self._vol_col]
        scaled = day_signals / (volatility + self._epsilon)

        # Limit to top N
        ranked = scaled.abs().sort_values(ascending=False)
        max_positions = int(self._max_lev / self._max_pos)
        top_symbols = ranked.head(max_positions).index
        top_scaled = scaled.loc[top_symbols]

        # Clip to limits
        clipped = top_scaled.clip(
            lower=-self._max_pos if self._allow_shorts else 0.0,
            upper=self._max_pos,
        )

        # Normalize to max_leverage
        total_abs = clipped.abs().sum()
        if total_abs > 0:
            clipped *= self._max_lev / total_abs

        # Rebuild MultiIndex series
        full_index = pd.MultiIndex.from_product(
            [[date], day_signals.index], names=["date", "symbol"]
        )
        adjusted = pd.Series(0.0, index=full_index, dtype=float)

        for symbol, weight in clipped.items():
            adjusted.loc[(date, symbol)] = weight

        return adjusted
