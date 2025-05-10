import random
from typing import Callable

import numpy as np
import pandas as pd

from trader.backtest.backtest_config import BacktestConfig
from trader.backtest.backtester import Backtester
from trader.core.engine import ModelBundle


def block_bootstrap(data: pd.DataFrame, block_size: int = 5) -> pd.DataFrame:
    """
    Resample the time series data by block bootstrap on dates.
    """
    dates = sorted(data["timestamp"].dt.floor("D").unique())
    num_blocks = len(dates) // block_size
    blocks = [dates[i * block_size : (i + 1) * block_size] for i in range(num_blocks)]
    sampled_blocks = random.choices(blocks, k=num_blocks)
    sampled_dates = [date for block in sampled_blocks for date in block]
    return data[data["timestamp"].dt.floor("D").isin(sampled_dates)].copy()


def symbol_subsample(data: pd.DataFrame, frac: float = 0.8) -> pd.DataFrame:
    """
    Randomly subsample a fraction of the symbols.
    """
    symbols = data["symbol"].unique()
    keep = np.random.choice(symbols, size=int(len(symbols) * frac), replace=False)
    return data[data["symbol"].isin(keep)].copy()


def run_robustness_test(
    data: pd.DataFrame,
    models: ModelBundle,
    config: BacktestConfig,
    num_trials: int = 50,
    block_size: int = 5,
    subsample_frac: float = 0.8,
    perturb_fn: Callable[[ModelBundle], ModelBundle] = lambda x: x,
) -> pd.DataFrame:
    """
    Run multiple randomized backtests to evaluate robustness.
    """
    results = []

    for i in range(num_trials):
        sampled_data = block_bootstrap(data, block_size=block_size)
        sampled_data = symbol_subsample(sampled_data, frac=subsample_frac)

        perturbed_models = perturb_fn(models)
        bt = Backtester(perturbed_models, config)
        trades, equity_curve = bt.run(sampled_data)

        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
        cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (
            252 / len(equity_curve)
        ) - 1
        volatility = equity_curve.pct_change().std() * np.sqrt(252)
        sharpe = cagr / volatility if volatility > 0 else 0
        max_dd = ((equity_curve / equity_curve.cummax()) - 1).min()

        results.append(
            {
                "trial": i,
                "total_return": total_return,
                "cagr": cagr,
                "sharpe": sharpe,
                "volatility": volatility,
                "max_drawdown": max_dd,
            }
        )

    return pd.DataFrame(results)
