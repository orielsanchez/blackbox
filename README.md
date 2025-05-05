# **BlackBox: Modular Backtesting Engine for Quantitative Trading**

**BlackBox** is a research-grade, extensible backtesting framework for building, testing, and evaluating multi-asset algorithmic trading strategies. It’s designed to simulate realistic market conditions, support modular model pipelines, and scale from lightweight experiments to full portfolio simulations.

---

## 🚀 Key Features

* 🔌 **Modular Architecture** — Plug-and-play support for custom alpha, risk, cost, slippage, execution, and portfolio models
* 📚 **Unified Multi-Symbol Simulation** — Allocate capital across thousands of assets in a single run
* ⚖️ **Realistic Execution & Costs** — Slippage and transaction cost models emulate real-world frictions
* 📊 **Comprehensive Metrics** — Auto-computes Sharpe, CAGR, max drawdown, win rate, turnover, and more
* 💾 **Efficient Data Access** — Fast Parquet loading using DuckDB and hive-style partitioned datasets
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
│   ├── data/               # Parquet data loaders (DuckDB)
│   ├── models/             # Alpha, risk, portfolio, etc.
│   ├── utils/              # Metrics, schema helpers
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
python src/trader/cli/run_backtest.py --universe universe/universe.csv
```

Or call it programmatically:

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

## 📂 Data Format

BlackBox expects Hive-partitioned Parquet data, e.g.:

```
data/ohlcv/minute/hive_parquet/
└── symbol=AAPL/
    └── date=2022-01-03/
        └── part-0.parquet
```

Each file should include:

* `timestamp`, `open`, `high`, `low`, `close`, `volume`, etc.

---

## 📊 Metrics Tracked

* **Portfolio metrics**: Sharpe, CAGR, volatility, drawdown
* **Trade metrics**: PnL, win rate, holding period, turnover
* **Execution metrics**: average slippage, fills, cost per trade

Results are logged to console and saved as:

* `results/{id}/trades.parquet`
* `results/{id}/equity.parquet`
* `results/{id}/metrics.json`

---

## 🗺️ Roadmap

* [x] Modular model interfaces
* [x] Multi-symbol backtesting and capital allocation
* [x] Execution, slippage, and cost tracking
* [x] Hive-style parquet loader with DuckDB
* [ ] Composite alpha blending and risk-adjusted scoring
* [ ] Streamlit dashboard for live result inspection
* [ ] Live/paper trading adapter (e.g., Alpaca, Binance)

---

## 📜 License

**MIT**

---

## ⚠️ Disclaimer

This software is provided for educational and research purposes only. It is not intended for live trading without thorough validation and risk controls. The authors assume no responsibility for any financial losses incurred.

---

> **BlackBox** bridges the gap between quant research and production-ready backtesting.
> *Trade like a quant. Test like a scientist.* 🧪📈
