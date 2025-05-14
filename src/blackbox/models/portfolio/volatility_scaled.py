from typing import Dict

import numpy as np
import pandas as pd

from blackbox.models.interfaces import PortfolioConstructionModel
from blackbox.utils.context import get_logger


class VolatilityScaledPortfolio(PortfolioConstructionModel):
    name = "volatility_scaled"

    def __init__(
        self,
        vol_lookback: int = 20,
        risk_target: float = 0.02,
        min_volatility: float = 1e-4,
        max_weight: float = 0.2,
        max_dollar_per_symbol: float = 100_000.0,
        min_notional: float = 1.0,
        min_price: float = 1.0,
    ):
        self.vol_lookback = vol_lookback
        self.risk_target = risk_target
        self.min_volatility = min_volatility
        self.max_weight = max_weight
        self.max_dollar_per_symbol = max_dollar_per_symbol
        self.min_notional = min_notional
        self.min_price = min_price

        self.logger = get_logger()

    def construct(self, alpha: pd.Series, snapshot: Dict) -> pd.Series:
        ohlcv: pd.DataFrame = snapshot.get("ohlcv")
        capital: float = snapshot.get("capital", 1_000_000.0)

        if capital <= 0:
            self.logger.warning("⚠️ Skipping portfolio construction: capital is 0")
            return pd.Series(dtype=float)

        if (
            not isinstance(ohlcv, pd.DataFrame)
            or ohlcv.empty
            or "close" not in ohlcv.columns
        ):
            self.logger.warning(
                "⚠️ Skipping portfolio construction: OHLCV missing or malformed"
            )
            return pd.Series(dtype=float)

        # 🔧 Collapse MultiIndex alpha to most recent day
        if isinstance(alpha.index, pd.MultiIndex):
            try:
                latest_date = alpha.index.get_level_values("date").max()
                alpha = alpha.xs(latest_date, level="date")
            except Exception:
                latest_date = alpha.index.get_level_values(0).max()
                alpha = alpha.xs(latest_date, level=0)

        # Deduplicate symbols
        if alpha.index.duplicated().any():
            dupes = alpha.index[alpha.index.duplicated()].tolist()
            self.logger.warning(f"⚠️ Duplicate symbols in alpha input: {dupes}")
            alpha = alpha.groupby(level=0).mean()

        # Normalize alpha
        if alpha.std() > 0:
            alpha = alpha / alpha.std()

        vols = {}
        for symbol in alpha.index:
            try:
                hist = ohlcv.xs(symbol, level="symbol")
                if len(hist) < self.vol_lookback:
                    continue
                returns = hist["close"].pct_change().dropna()
                vol = returns[-self.vol_lookback :].std()
                if pd.notna(vol):
                    vols[symbol] = max(vol, self.min_volatility)
            except KeyError:
                self.logger.warning(f"⚠️ No OHLCV data for {symbol}")
                continue

        if not vols:
            self.logger.warning("⚠️ No volatilities computed — skipping day.")
            return pd.Series(dtype=float)

        vol_series = pd.Series(vols)

        notional_risk = self.risk_target * capital
        dollar_targets = (notional_risk / vol_series).clip(
            upper=self.max_dollar_per_symbol
        )

        common = alpha.index.intersection(dollar_targets.index)
        if common.empty:
            return pd.Series(dtype=float)

        try:
            recent_prices = (
                ohlcv.loc[(slice(None), common), "close"].groupby("symbol").last()
            )
            common = common.intersection(
                recent_prices[recent_prices >= self.min_price].index
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to filter by price: {e}")

        if common.empty:
            self.logger.warning("⚠️ All symbols filtered out by price")
            return pd.Series(dtype=float)

        dollar_exposures = alpha[common] * dollar_targets[common]
        weights = dollar_exposures / capital

        weights = weights.clip(lower=-self.max_weight, upper=self.max_weight)
        weights = weights[abs(weights * capital) >= self.min_notional]

        if weights.index.duplicated().any():
            dupes = weights.index[weights.index.duplicated()].tolist()
            self.logger.warning(f"⚠️ Duplicate symbols in portfolio output: {dupes}")
            weights = weights.groupby(level=0).sum()

        gross_exposure = weights.abs().sum()
        if gross_exposure > 0:
            weights *= min(1.0, 1.0 / gross_exposure)

        self.logger.info(
            f"✅ Constructed {len(weights)} positions | Gross exposure: {gross_exposure:.2f}"
        )
        return weights.fillna(0).sort_index()


def mark_to_market(self, prices: pd.Series):
    portfolio_val = 0.0
    for symbol, weight in self.invested_weights.items():
        if symbol in prices:
            portfolio_val += weight * prices[symbol]
    self.portfolio_value = portfolio_val
