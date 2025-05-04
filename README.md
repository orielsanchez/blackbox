# BlackBox: Modular Backtesting Framework for Algorithmic Trading

**BlackBox** is a modular, research-grade backtesting engine designed for developing, evaluating, and deploying multi-asset trading strategies. It supports the full lifecycle from alpha modeling to execution simulation with extensible interfaces and realistic market conditions.

## Features

* 🔌 **Pluggable Models**: Modular architecture for alpha, risk, transaction cost, portfolio, execution, and slippage models
* 🧠 **Multi-symbol Support**: Runs unified capital allocation across hundreds or thousands of assets
* 📈 **Realistic Execution**: Slippage and transaction cost simulation for institutional-grade realism
* 📊 **Performance Metrics**: Computes Sharpe, drawdown, volatility, CAGR, PnL, win rate, and execution stats
* 💾 **Hive-style Parquet Ingestion**: Efficient loading of minute-level data from local files using DuckDB
* ⚙️ **Backtest Configs**: Declarative configuration for backtest parameters and model tuning
* 📦 **Built for Scale**: Designed to grow from small experiments to portfolio-level optimization

## Project Structure

```
blackbox/
├── src/
│   └── trader/
│       ├── backtest/              # Backtest config and engine
│       │   ├── backtest_config.py
│       │   └── backtester.py
│       ├── cli/                   # Entry points and run scripts
│       │   └── run_backtester.py
│       ├── core/                  # Interfaces for engine components
│       │   ├── alpha.py
│       │   ├── engine.py
│       │   ├── execution.py
│       │   ├── portfolio.py
│       │   ├── risk.py
│       │   ├── slippage.py
│       │   └── tx_cost.py
│       ├── dashboard/             # Dashboard interface (future)
│       │   └── backtest_dashboard.py
│       ├── data/                  # Local parquet data clients
│       │   ├── parquet_data_client.py
│       │   └── parquet_data_loader.py
│       ├── models/                # Model implementations
│       │   ├── alpha/
│       │   │   ├── momentum.py
│       │   │   └── mean_reversion.py
│       │   ├── execution/
│       │   │   └── simple.py
│       │   ├── portfolio/
│       │   │   └── equal_weight.py
│       │   ├── risk/
│       │   │   └── ewma.py
│       │   ├── slippage/
│       │   │   └── percent.py
│       │   └── tx_cost/
│       │       └── fixed.py
│       └── utils/
│           ├── metrics.py
│           └── __init__.py
├── tests/                         # Unit tests
├── notebooks/                     # Exploratory analysis
├── results/                       # Saved output from backtests
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

## Example Usage

```bash
python src/trader/cli/run_backtester.py
```

Or integrate into a script:

```python
from trader.backtest.backtester import Backtester
from trader.utils.metrics import calculate_performance

backtester = Backtester(models, config)
equity_curve = backtester.run(data)
metrics = calculate_performance(equity_curve)
```

## Data Format

Parquet files are stored in Hive-style partitions:

```
data/ohlcv/minute/hive_parquet/
└── symbol=AAPL/
    └── date=2022-01-03/
        └── part-0.parquet
```

Each file should contain columns like `timestamp`, `open`, `high`, `low`, `close`, `volume`, etc.

## Metrics

After each run, the system computes and logs:

* Portfolio stats: Sharpe, CAGR, drawdown, volatility
* Trade stats: PnL, win rate, average trade size
* Execution stats: average slippage (bps), total slippage, number of fills

## Roadmap

* [x] Alpha + Risk + Cost + Portfolio + Execution model pipeline
* [x] Hive-partitioned local data pipeline with DuckDB
* [x] Slippage tracking and execution stats
* [ ] Paper/live trading interface (e.g., Alpaca adapter)
* [ ] Streamlit dashboard for visualization
* [ ] Composite alpha blending and feature engineering

## License

MIT

## Disclaimer

This software is provided for educational and research purposes only. It is not intended for use in live trading without rigorous validation. The authors and contributors take no responsibility for financial losses or decisions made using this software.

---

**BlackBox** is built for fast iteration, research reproducibility, and bridging the gap between quant prototypes and production-grade trading systems.

> Trade like a quant, test like a scientist. 🧪📈
