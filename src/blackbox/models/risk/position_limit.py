import pandas as pd

from blackbox.models.interfaces import RiskModel


class PositionLimitRisk(RiskModel):
    def __init__(
        self,
        max_leverage: float = 1.0,
        max_position_size: float = 0.25,
        allow_shorts: bool = True,
    ):
        """
        Risk model to cap individual position sizes and enforce portfolio leverage limits.

        Args:
            max_leverage: Total portfolio leverage allowed (e.g. 1.0 = 100%).
            max_position_size: Maximum allowed weight per position (e.g. 0.25 = 25%).
            allow_shorts: If False, all weights will be clipped to [0, max_position_size].
        """
        self._max_leverage = max_leverage
        self._max_position_size = max_position_size
        self._allow_shorts = allow_shorts

    @property
    def name(self) -> str:
        return "position_limit"

    def apply(self, signals: pd.Series, features: pd.DataFrame) -> pd.Series:
        if signals.empty:
            return pd.Series(dtype=float)

        # 1️⃣ Extract current date from MultiIndex
        dates = signals.index.get_level_values("date").unique()
        if len(dates) != 1:
            raise ValueError(f"RiskModel expects one date per call, got: {dates}")
        current_date = dates[0]

        # 2️⃣ Flatten to symbol-level
        day_signals = signals.loc[current_date]

        # 3️⃣ Rank top N by abs(signal)
        ranked = day_signals.abs().sort_values(ascending=False)
        max_positions = int(self._max_leverage / self._max_position_size)
        top_symbols = ranked.head(max_positions).index

        # 4️⃣ Grab real signal values for these symbols
        top_signals = day_signals.loc[top_symbols]

        # 5️⃣ Clip and normalize
        clipped = top_signals.clip(
            lower=-self._max_position_size if self._allow_shorts else 0.0,
            upper=self._max_position_size,
        )

        total_abs = clipped.abs().sum()

        if total_abs > 0:
            clipped *= self._max_leverage / total_abs

        # 6️⃣ Construct MultiIndex output
        full_index = pd.MultiIndex.from_product(
            [[current_date], day_signals.index], names=["date", "symbol"]
        )
        adjusted = pd.Series(0.0, index=full_index, dtype=float)

        for sym, weight in clipped.items():
            adjusted.loc[(current_date, sym)] = weight

        return adjusted
