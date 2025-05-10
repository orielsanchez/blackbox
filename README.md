# **BlackBox: Modular Backtesting Engine for Quantitative Trading**

**BlackBox** is a research-grade, extensible backtesting framework for building, testing, and evaluating multi-asset algorithmic trading strategies. It’s designed to simulate realistic market conditions, support modular model pipelines, and scale from lightweight experiments to full portfolio simulations.

---

## 🚀 Key Features

* 🔌 **Modular Architecture** — Plug-and-play support for custom alpha, risk, cost, slippage, execution, and portfolio models
* 📚 **Unified Multi-Symbol Simulation** — Allocate capital across thousands of assets in a single run
* ⚖️ **Realistic Execution & Costs** — Slippage, transaction costs, and partial fills modeled with real-world constraints
* 📊 **Comprehensive Metrics** — Auto-computes Sharpe, CAGR, max drawdown, win rate, turnover, and more
* 📀 **DuckDB-Powered Data Access** — Efficient partitioned Parquet querying via DuckDB
* 🔀 **Monte Carlo Testing** — Stress-test strategies with randomized subsampling, block bootstraps, and parameter perturbations
* ⚙️ **Config-Driven Runs** — Backtests are easily controlled via declarative `BacktestConfig` objects
* 🧪 **Built for Experimentation** — Log every trade, equity curve, and metric for reproducibility and analysis

---

## 🧱 Project Structure

```
blackbox/
├── src/trader/
│   ├── backtest/           # Backtest engine and config
│   ├── cli/                # CLI scripts and entry points
│   ├── core/               # Abstract model interfaces
│   ├── data/               # DuckDB-based parquet loaders
│   ├── models/             # Alpha, risk, portfolio, execution, tx_cost, slippage
│   ├── utils/              # Metrics, diagnostics, resampling
├── universe/               # Symbol universe definitions
├── results/                # Saved trades, equity, and metrics
├── notebooks/              # Exploratory notebooks
├── tests/                  # Unit tests
├── README.md
├── pyproject.toml
```

---

## 🧪 Example: Run a Backtest

```bash
python src/trader/cli/run_backtest.py --universe universe/universe.csv --duckdb_path db/ohlcv.duckdb
```

Or use it programmatically:

```python
from trader.backtest.backtester import Backtester
from trader.utils.metrics import calculate_performance

backtester = Backtester(models, config)
result_df, equity_curve = backtester.run(data)
metrics = calculate_performance(equity_curve)
```

After the run:

* Trades, equity curve, and metrics are saved to `results/{backtest_id}/`

---

## 🔀 Example: Run Monte Carlo Robustness

```bash
python src/trader/cli/run_monte_backtest.py --universe universe/universe.csv --duckdb_path db/ohlcv.duckdb --trials 100
```

This runs 100 randomized trials with block bootstrapping, symbol subsampling, and noise injection. Output includes summary stats and trial-by-trial metrics in Parquet.

---

## 📂 Data Format

BlackBox uses Hive-partitioned Parquet format with DuckDB:

```
data/ohlcv/minute/hive_parquet/
└── symbol=AAPL/
    └── date=2022-01-03/
        └── part-0.parquet
```

Each file must contain:

* `timestamp`, `open`, `high`, `low`, `close`, `volume`, etc.

---

## 📊 Metrics Tracked

* **Portfolio**: Sharpe, CAGR, volatility, drawdown
* **Trade**: PnL, win rate, turnover, holding period
* **Execution**: average slippage, partial fills, cost per trade

Saved to:

* `results/{id}/trades.parquet`
* `results/{id}/equity.parquet`
* `results/{id}/metrics.json`

---

## 🔧 Realistic Models

* `VolumeAdjustedTxCostModel`: cost scales with notional and liquidity
* `VolatilityImpactSlippageModel`: slippage linked to EWMA volatility
* `DelayedPartialFillExecutionModel`: simulates partial fills and execution delays

These models increase realism for stress testing execution and performance.

---

## 🗺️ Roadmap

* [x] Modular model interfaces
* [x] Multi-symbol backtesting and capital allocation
* [x] Realistic cost and slippage simulation
* [x] Monte Carlo resampling for robustness
* [ ] Composite alpha blending with dynamic regime switching
* [ ] Streamlit dashboard for visualization
* [ ] Live/paper trading via broker adapters (e.g., Alpaca)

---

## 📜 License

**MIT**

---

## ⚠️ Disclaimer

This software is provided for educational and research purposes only. It is not intended for live trading without thorough validation and risk controls. The authors assume no responsibility for any financial losses incurred.

---

> **BlackBox** bridges the gap between quant research and production-ready backtesting.
> *Trade like a quant. Test like a scientist.* 🧪📈
