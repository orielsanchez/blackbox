from typing import Dict

import pandas as pd

from blackbox.core.types.context import OHLCVSnapshot
from blackbox.core.types.types import PortfolioTarget
from blackbox.models.interfaces import PortfolioConstructionModel
from blackbox.utils.context import get_logger
from blackbox.utils.normalization import SignalNormalizer


class VolatilityScaledPortfolio(PortfolioConstructionModel):
    ALLOWED_NORMALIZATIONS = {
        "zscore",
        "minmax",
        "rank",
        "softmax",
        "winsorized_zscore",
    }

    def __init__(
        self,
        vol_lookback: int = 20,
        risk_target: float = 0.02,
        min_volatility: float = 1e-4,
        max_weight: float = 0.2,
        max_dollar_per_symbol: float = 100_000.0,
        min_notional: float = 1.0,
        min_price: float = 1.0,
        normalization: str = "zscore",
        max_turnover: float = 1.0,
        max_trades_per_day: int = 9999,
    ):
        self.vol_lookback = vol_lookback
        self.risk_target = risk_target
        self.min_volatility = min_volatility
        self.max_weight = max_weight
        self.max_dollar_per_symbol = max_dollar_per_symbol
        self.min_notional = min_notional
        self.min_price = min_price
        self.normalization = normalization
        self.max_turnover = max_turnover
        self.max_trades_per_day = max_trades_per_day

        self.logger = get_logger()
        self.invested_weights = {}
        self.portfolio_value = 0.0

        if normalization not in self.ALLOWED_NORMALIZATIONS:
            raise ValueError(f"❌ Unsupported normalization method: {normalization}")

    @property
    def name(self) -> str:
        return "volatility_scaled"

    def normalize_signals(self, alpha: pd.Series) -> pd.Series:
        method = getattr(SignalNormalizer, self.normalization, None)
        if method is None:
            self.logger.warning(f"⚠️ Unknown normalization method: {self.normalization}")
            return alpha

        if alpha.std() == 0 or alpha.nunique() <= 1:
            self.logger.warning("Cannot normalize signals - all values are identical")
            alpha = alpha.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
            return alpha

        try:
            normalized = method(alpha)
            if normalized.isna().any():
                self.logger.warning(
                    "Normalization produced NaN values, using original signals"
                )
                return alpha
            return normalized
        except Exception as e:
            self.logger.warning(f"Error in normalization: {e}. Using original signals.")
            return alpha

    def construct(
        self,
        signals: pd.Series,
        capital: float,
        features,
        snapshot: OHLCVSnapshot,
    ) -> PortfolioTarget:
        date = snapshot.date
        self.logger.debug(
            f"🔧 Portfolio construction with {len(signals)} signals and ${capital:.2f} capital"
        )

        if isinstance(signals.index, pd.MultiIndex):
            signals = signals.xs(date, level="date")

        if signals.empty or capital <= 0:
            self.logger.warning(
                "Skipping portfolio construction: empty signals or non-positive capital"
            )
            return PortfolioTarget(
                date=date, weights=pd.Series(dtype=float), capital=capital
            )

        if signals.index.duplicated().any():
            signals = signals.groupby(level=0).mean()

        signals = self.normalize_signals(signals)
        signals = signals.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

        # ─── Estimate volatility ───
        vol_by_symbol: Dict[str, float] = {}
        for symbol in signals.index:
            try:
                if symbol in snapshot.other:
                    df = snapshot.other[symbol].sort_index().tail(self.vol_lookback + 1)
                    returns = df.pct_change().dropna()
                    vol = returns.std()
                    vol_by_symbol[symbol] = (
                        max(vol, self.min_volatility)
                        if pd.notna(vol)
                        else self.min_volatility
                    )
                else:
                    raise KeyError("No extra data for symbol")
            except Exception as e:
                self.logger.debug(f"[{symbol}] volatility fallback: {e}")
                vol_by_symbol[symbol] = self.min_volatility

        vol_series = pd.Series(vol_by_symbol)
        notional_risk = self.risk_target * capital
        dollar_targets = (notional_risk / vol_series).clip(
            upper=self.max_dollar_per_symbol
        )

        price_snapshot = snapshot.close
        valid_symbols = price_snapshot[
            price_snapshot >= self.min_price
        ].index.intersection(signals.index)

        if valid_symbols.empty:
            self.logger.error(f"No valid symbols after price filter on {date}")
            return PortfolioTarget(
                date=date, weights=pd.Series(dtype=float), capital=capital
            )

        dollar_exposures = signals[valid_symbols] * dollar_targets[valid_symbols]
        weights = dollar_exposures / capital
        weights = weights.clip(lower=-self.max_weight, upper=self.max_weight)

        notional_values = (weights * capital).abs()
        weights = weights[notional_values >= self.min_notional]

        if weights.empty:
            self.logger.warning("No positions pass min_notional filter")
            return PortfolioTarget(
                date=date, weights=pd.Series(dtype=float), capital=capital
            )

        if weights.index.duplicated().any():
            weights = weights.groupby(level=0).sum()

        gross_exposure = weights.abs().sum()
        if gross_exposure > 0:
            weights *= min(1.0, 1.0 / gross_exposure)

        # ─── Enforce turnover cap ───
        prev_weights = (
            pd.Series(self.invested_weights, dtype=float)
            .reindex(weights.index)
            .fillna(0.0)
        )
        turnover = (weights - prev_weights).abs().sum()

        if turnover > self.max_turnover:
            scale = self.max_turnover / turnover
            self.logger.debug(
                f"⚠️ Turnover {turnover:.4f} > max {self.max_turnover:.4f}, scaling by {scale:.4f}"
            )
            weights *= scale

        # ─── Enforce max trades per day ───
        trades = (weights - prev_weights).abs()
        num_trades = (trades > 1e-6).sum()

        if num_trades > self.max_trades_per_day:
            self.logger.debug(
                f"⚠️ {num_trades} trades exceeds max {self.max_trades_per_day}, keeping largest"
            )
            keep = (
                trades.sort_values(ascending=False).head(self.max_trades_per_day).index
            )
            weights = weights[keep]

        self.invested_weights = weights.to_dict()
        self.logger.debug(f"✅ Final portfolio: {self.invested_weights}")
        return PortfolioTarget(date=date, weights=weights.sort_index(), capital=capital)

    def feedback_from_execution(self, feedback: Dict[str, Dict[str, object]]) -> None:
        pass

    def mark_to_market(self, prices: pd.Series) -> None:
        self.portfolio_value = sum(
            self.invested_weights.get(symbol, 0.0) * prices.get(symbol, 0.0)
            for symbol in self.invested_weights
        )
