import pandas as pd

from blackbox.models.interfaces import RiskModel


class VolatilityCapRisk(RiskModel):
    """
    Risk model that filters or downweights symbols exceeding a volatility cap.
    Supports multi-date signals via groupby on date index level.
    """

    def __init__(
        self,
        max_volatility: float = 0.05,
        volatility_column: str = "rolling_vol_20",
        allow_shorts: bool = True,
    ):
        self._max_volatility = max_volatility
        self._volatility_column = volatility_column
        self._allow_shorts = allow_shorts

    @property
    def name(self) -> str:
        return "volatility_cap"

    def apply(self, signals: pd.Series, features: pd.DataFrame) -> pd.Series:
        if signals.empty:
            return pd.Series(dtype=float)

        all_dates = signals.index.get_level_values("date").unique()
        output: list[pd.Series] = []

        for date in all_dates:
            try:
                daily_signals = signals.loc[date]
                if isinstance(daily_signals, pd.DataFrame):
                    daily_signals = daily_signals.squeeze()
                daily_features = features.loc[date]
            except KeyError:
                continue

            if self._volatility_column not in daily_features.columns:
                raise ValueError(
                    f"Missing column '{self._volatility_column}' in features"
                )

            vol = daily_features[self._volatility_column]
            valid_symbols = vol[vol <= self._max_volatility].index

            filtered = daily_signals.loc[
                daily_signals.index.intersection(valid_symbols)
            ]
            if isinstance(filtered, pd.DataFrame):
                filtered = filtered.squeeze()

            clipped = filtered.clip(
                lower=-1.0 if self._allow_shorts else 0.0,
                upper=1.0,
            )

            total_abs = clipped.abs().sum()
            if total_abs > 0:
                clipped *= 1.0 / total_abs

            clipped.index = pd.MultiIndex.from_product(
                [[date], clipped.index], names=["date", "symbol"]
            )
            output.append(clipped)

        if output:
            result = pd.concat(output)
        else:
            result = pd.Series(dtype=float)

        return result.reindex(signals.index, fill_value=0.0)
