from typing import Dict

import numpy as np
import pandas as pd

from blackbox.models.interfaces import PortfolioConstructionModel


class VolatilityScaledPortfolio(PortfolioConstructionModel):
    name = "volatility_scaled"

    def __init__(
        self,
        vol_lookback: int = 20,
        risk_target: float = 0.02,
        min_volatility: float = 1e-4,
        max_weight: float = 0.2,
    ):
        self.vol_lookback = vol_lookback
        self.risk_target = risk_target
        self.min_volatility = min_volatility
        self.max_weight = max_weight

    def construct(self, alpha: pd.Series, snapshot: Dict) -> pd.Series:
        ohlcv: Dict[str, pd.DataFrame] = snapshot["ohlcv"]
        vols = {}

        for symbol in alpha.index:
            hist = ohlcv.get(symbol)
            if hist is None or len(hist) < self.vol_lookback:
                continue
            ret = hist["close"].pct_change().dropna()
            vol = ret[-self.vol_lookback :].std()
            if pd.notna(vol):
                vols[symbol] = max(vol, self.min_volatility)

        if not vols:
            return pd.Series(dtype=float)

        vol_series = pd.Series(vols)
        raw_weights = self.risk_target / vol_series  # Each asset gets ~2% risk

        scaled = alpha.loc[raw_weights.index] * raw_weights

        # Normalize portfolio
        if scaled.abs().sum() > 0:
            scaled = scaled / scaled.abs().sum()

        scaled = scaled.clip(lower=-self.max_weight, upper=self.max_weight)
        return scaled
