from typing import Dict

import pandas as pd

from blackbox.models.interfaces import PortfolioConstructionModel
from blackbox.utils.context import get_logger
from blackbox.utils.normalization import SignalNormalizer


class VolatilityScaledPortfolio(PortfolioConstructionModel):
    name = "volatility_scaled"

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
    ):
        self.vol_lookback = vol_lookback
        self.risk_target = risk_target
        self.min_volatility = min_volatility
        self.max_weight = max_weight
        self.max_dollar_per_symbol = max_dollar_per_symbol
        self.min_notional = min_notional
        self.min_price = min_price
        self.normalization = normalization
        self.logger = get_logger()

        if normalization not in self.ALLOWED_NORMALIZATIONS:
            raise ValueError(f"❌ Unsupported normalization method: {normalization}")

        self.invested_weights = {}
        self.portfolio_value = 0.0

    def normalize_signals(self, alpha: pd.Series) -> pd.Series:
        method = getattr(SignalNormalizer, self.normalization, None)
        if method is None:
            self.logger.warning(f"⚠️ Unknown normalization method: {self.normalization}")
            return alpha

        # Check for all identical values
        if alpha.std() == 0:
            self.logger.warning("Cannot normalize signals - all values are identical")
            return alpha

        try:
            normalized = method(alpha)
            # Check for NaN values
            if normalized.isna().any():
                self.logger.warning("Normalization produced NaN values, using original signals")
                return alpha
            return normalized
        except Exception as e:
            self.logger.warning(f"Error in normalization: {e}. Using original signals.")
            return alpha

    def construct(self, alpha: pd.Series, snapshot: Dict) -> pd.Series:
        ohlcv: pd.DataFrame = snapshot.get("ohlcv")
        capital: float = snapshot.get("capital", 1_000_000.0)

        # DEBUG: Log input parameters
        self.logger.debug(
            f"Portfolio construction input: alpha={len(alpha)}, capital=${capital:.2f}"
        )

        if alpha.empty:
            self.logger.warning("Empty alpha signals, skipping portfolio construction")
            return pd.Series(dtype=float)

        self.logger.debug(f"Alpha signals range: min={alpha.min():.6f}, max={alpha.max():.6f}")

        if capital <= 0:
            self.logger.warning("⚠️ Skipping portfolio construction: capital is 0")
            return pd.Series(dtype=float)

        if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty or "close" not in ohlcv.columns:
            self.logger.warning("⚠️ Skipping portfolio construction: OHLCV missing or malformed")
            self.logger.debug(
                f"ohlcv type: {type(ohlcv)}, empty: {getattr(ohlcv, 'empty', True)}, columns: {getattr(ohlcv, 'columns', None)}"
            )
            return pd.Series(dtype=float)

        # 🔧 Collapse MultiIndex alpha to most recent day
        if isinstance(alpha.index, pd.MultiIndex):
            try:
                latest_date = alpha.index.get_level_values("date").max()
                alpha = alpha.xs(latest_date, level="date")
                self.logger.debug(f"Collapsed MultiIndex from alpha, new shape: {alpha.shape}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to collapse MultiIndex: {e}")
                try:
                    latest_date = alpha.index.get_level_values(0).max()
                    alpha = alpha.xs(latest_date, level=0)
                    self.logger.debug(
                        f"Collapsed MultiIndex from level 0, new shape: {alpha.shape}"
                    )
                except Exception as e2:
                    self.logger.error(f"❌ Failed all MultiIndex handling attempts: {e2}")
                    return pd.Series(dtype=float)

        # Deduplicate symbols
        if alpha.index.duplicated().any():
            dupes = alpha.index[alpha.index.duplicated()].tolist()
            self.logger.warning(f"⚠️ Duplicate symbols in alpha input: {dupes}")
            alpha = alpha.groupby(level=0).mean()
            self.logger.debug(f"After deduplication, alpha shape: {alpha.shape}")

        # Normalize alpha
        alpha = self.normalize_signals(alpha)
        self.logger.debug(
            f"After normalization, alpha range: {alpha.min():.4f} to {alpha.max():.4f}"
        )

        # IMPORTANT DEBUG POINT: Check OHLCV structure
        self.logger.debug(f"OHLCV index names: {ohlcv.index.names}")
        self.logger.debug(f"OHLCV sample shapes: rows={len(ohlcv)}, columns={len(ohlcv.columns)}")

        # Check actual alpha symbols against OHLCV symbols
        ohlcv_symbols = set()
        if isinstance(ohlcv.index, pd.MultiIndex):
            ohlcv_symbols = set(ohlcv.index.get_level_values(1))
        else:
            ohlcv_symbols = set(ohlcv.index)

        alpha_symbols = set(alpha.index)
        common_symbols = alpha_symbols.intersection(ohlcv_symbols)
        self.logger.debug(
            f"Symbol overlap: {len(common_symbols)}/{len(alpha_symbols)} alpha symbols found in OHLCV data"
        )

        # Calculate volatilities for each symbol - FIX for the index issue
        try:
            # Extract current date's price data
            prices_today = snapshot.get("prices")

            # Manually calculate volatilities for exactly the symbols in alpha
            vols = {}

            for symbol in alpha.index:
                try:
                    # Extract price history for this symbol
                    if isinstance(ohlcv.index, pd.MultiIndex):
                        symbol_data = ohlcv.xs(symbol, level=1, drop_level=False)

                        # Get the most recent sequence of data
                        if len(symbol_data) > self.vol_lookback:
                            symbol_data = symbol_data.sort_index().tail(self.vol_lookback + 1)

                        # Calculate returns and volatility
                        returns = symbol_data["close"].pct_change().dropna()
                        vol = returns.std()

                        if pd.notna(vol) and vol > 0:
                            vols[symbol] = max(vol, self.min_volatility)
                        else:
                            vols[symbol] = self.min_volatility

                    else:
                        # If OHLCV is not MultiIndex, use a fallback approach
                        vols[symbol] = self.min_volatility

                except Exception as e:
                    self.logger.debug(f"Error calculating volatility for {symbol}: {e}")
                    vols[symbol] = self.min_volatility

            vol_series = pd.Series(vols)
            self.logger.debug(
                f"Manual volatility calculation: {len(vol_series)} symbols, range: {vol_series.min():.6f} to {vol_series.max():.6f}"
            )

        except Exception as e:
            self.logger.error(f"Failed to calculate volatilities: {e}")
            # Last resort - use default volatility for all symbols
            vol_series = pd.Series(self.min_volatility, index=alpha.index)
            self.logger.debug(
                f"Using default volatility {self.min_volatility} for all {len(vol_series)} symbols"
            )

        # Calculate dollar risk allocations
        notional_risk = self.risk_target * capital
        self.logger.debug(
            f"Notional risk: ${notional_risk:.2f} (risk_target={self.risk_target}, capital=${capital:.2f})"
        )

        dollar_targets = (notional_risk / vol_series).clip(upper=self.max_dollar_per_symbol)
        self.logger.debug(
            f"Dollar targets: min=${dollar_targets.min():.2f}, max=${dollar_targets.max():.2f}, count={len(dollar_targets)}"
        )

        # Find common symbols between alpha and vol calculations
        common = alpha.index.intersection(vol_series.index)
        self.logger.debug(
            f"Common symbols after vol calculation: {len(common)} (vs alpha: {len(alpha)}, vol: {len(vol_series)})"
        )

        if common.empty:
            self.logger.error("❌ No overlap between alpha signals and volatility targets")
            return pd.Series(dtype=float)

        # Apply price filter
        try:
            # Get prices from snapshot
            prices_from_snapshot = snapshot.get("prices")
            if prices_from_snapshot is not None and not prices_from_snapshot.empty:
                self.logger.debug(f"Using {len(prices_from_snapshot)} prices from snapshot")
                common_prices = common.intersection(prices_from_snapshot.index)

                if len(common_prices) == 0:
                    # If no common symbols, create a log to diagnose
                    self.logger.error(
                        f"No common symbols between alpha/vol and prices! Alpha/vol symbols: {list(common)[:5]}, Price symbols: {list(prices_from_snapshot.index)[:5]}"
                    )
                    # Use all common symbols anyway
                    filtered_common = common
                else:
                    # Filter by price
                    price_filtered = prices_from_snapshot[common_prices]
                    price_filtered = price_filtered[price_filtered >= self.min_price]
                    filtered_common = price_filtered.index
                    self.logger.debug(
                        f"Price filter: {len(filtered_common)}/{len(common_prices)} symbols pass min_price=${self.min_price}"
                    )
            else:
                self.logger.warning("No prices in snapshot, skipping price filter")
                filtered_common = common

        except Exception as e:
            self.logger.error(f"Error in price filtering: {e}")
            filtered_common = common

        if len(filtered_common) == 0:
            self.logger.error("❌ No symbols left after price filtering")
            return pd.Series(dtype=float)

        # Calculate positions
        dollar_exposures = alpha[filtered_common] * dollar_targets[filtered_common]
        self.logger.debug(
            f"Dollar exposures: min=${dollar_exposures.min():.2f}, max=${dollar_exposures.max():.2f}, count={len(dollar_exposures)}"
        )

        weights = dollar_exposures / capital
        self.logger.debug(f"Initial weights: min={weights.min():.6f}, max={weights.max():.6f}")

        weights = weights.clip(lower=-self.max_weight, upper=self.max_weight)
        self.logger.debug(
            f"After clipping to max_weight={self.max_weight}: min={weights.min():.6f}, max={weights.max():.6f}"
        )

        # Add debug info about notional filtering
        before_filter = len(weights)
        notional_values = weights.abs() * capital
        below_min = weights[notional_values < self.min_notional]
        if not below_min.empty:
            self.logger.debug(
                f"Filtering out {len(below_min)} positions below min notional (${self.min_notional})"
            )
            self.logger.debug(
                f"Notional values filtered: {notional_values[below_min.index].to_dict()}"
            )

        weights = weights[notional_values >= self.min_notional]
        self.logger.debug(f"After notional filter: {before_filter} → {len(weights)} positions")

        if weights.empty:
            self.logger.warning("No positions left after notional filter")
            return pd.Series(dtype=float)

        if weights.index.duplicated().any():
            dupes = weights.index[weights.index.duplicated()].tolist()
            self.logger.warning(f"⚠️ Duplicate symbols in portfolio output: {dupes}")
            weights = weights.groupby(level=0).sum()

        gross_exposure = weights.abs().sum()
        self.logger.debug(f"Gross exposure before scaling: {gross_exposure:.4f}")

        if gross_exposure > 0:
            scale_factor = min(1.0, 1.0 / gross_exposure)
            weights *= scale_factor
            self.logger.debug(f"Applied scaling factor: {scale_factor:.4f}")

        self.invested_weights = weights.to_dict()
        self.logger.debug(f"Final weights: {weights.to_dict()}")
        self.logger.info(
            f"✅ Constructed {len(weights)} positions | Gross exposure: {weights.abs().sum():.4f}"
        )
        return weights.fillna(0).sort_index()

    def feedback_from_execution(self, feedback: dict):
        pass  # Not implemented yet

    def mark_to_market(self, prices: pd.Series):
        portfolio_val = 0.0
        for symbol, weight in self.invested_weights.items():
            if symbol in prices:
                portfolio_val += weight * prices[symbol]
        self.portfolio_value = portfolio_val
