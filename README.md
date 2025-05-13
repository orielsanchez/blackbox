
# BlackBox Backtesting Framework

A modular, extensible backtesting engine for quantitative trading strategies. Built for research and production-aligned simulation. Components are plug-and-play and configured via YAML.

## 📦 Features

- Modular architecture: alpha, risk, cost, portfolio, execution models
- YAML-driven strategy configuration
- DuckDB-based historical data backend
- Slippage, commission, transaction cost simulation
- Risk control: leverage, position sizing, short constraints
- Diagnostics logging at each step: signals, trades, portfolios
- Performance analytics (Sharpe, max drawdown, annualized return)
- Extensible via registries and model inheritance

## 🗂️ Project Structure

```
blackbox/
│
├── cli/                    # Command-line tools
│   └── main.py             # Entry point for backtesting
│
├── config/                 # Typed dataclasses and schema
│   └── schema.py
│
├── data/                   # Data access layer
│   └── loader.py           # DuckDBDailyLoader
│
├── models/                 # Alpha, risk, cost, portfolio, execution models
│   ├── base.py             # BaseModel API
│   ├── registry.py         # Model discovery/registry
│   └── alpha/mean_reversion.py
│
├── metrics/                # Performance calculations
│   └── performance.py      # Sharpe, drawdown, equity
│
├── utils/                  # Logging, helpers, etc.
│   ├── logger.py           # Rich console+file logger
│
├── results/                # Output directory for each run
└── universe/               # CSV files of tickers per strategy
```

## ⚙️ Installation

```bash
git clone https://github.com/orielsanchez/blackbox.git
cd blackbox
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🧠 Example Usage

### Run a Backtest

```bash
python src/blackbox/cli/main.py --config config/strategies/debug_test.yaml
```

### Example Config

```yaml
run_id: debug_microtest
start_date: 2022-12-01
end_date: 2023-01-10
universe_file: universe/micro_universe.csv

data:
  db_path: db/ohlcv.duckdb
  rolling: true
  window: 5

alpha_model:
  name: mean_reversion
  params:
    window: 5
    threshold: 0.5

risk_model:
  name: position_limit
  params:
    max_leverage: 1.0
    max_position_size: 0.5
    allow_shorts: false

tx_cost_model:
  name: quadratic_market_impact
  params:
    commission_rate: 0.0001
    impact_coefficient: 5e-5
    min_commission: 1.0

portfolio_model:
  name: volatility_scaled
  params:
    vol_lookback: 5
    risk_target: 0.01
    max_weight: 0.4

execution_model:
  name: market
  params:
    slippage: 0.0001
    commission: 0.0001
    initial_portfolio_value: 100000
    fractional: true
    allow_shorts: false
    min_notional: 10.0

min_holding_period: 2
settlement_delay: 1
log_level: DEBUG
log_to_console: true
log_to_file: true
```

## 📊 Metrics Output

After a run, the system saves:

- Equity curve (NAV)
- Daily trades
- Model signals
- Final metrics:
  - Total Return (%)
  - Annualized Return (%)
  - Annualized Volatility (%)
  - Sharpe Ratio
  - Max Drawdown (%)

Output saved in `results/<run_id>/`.

## 📁 Data Format

OHLCV data is loaded via DuckDB from a table with structure:

```
symbol | day | open | high | low | close | volume
```

Universe files are simple CSVs:

```
symbol
AAPL
GOOG
MSFT
```

## 🔧 Extending

Add new models by subclassing `BaseModel` and registering:

```python
@register_model("my_alpha")
class MyAlpha(BaseModel):
    ...
```

Models are auto-discovered and loaded based on the config file.

## 📝 License

MIT License

## ✨ Credits

- Built with `rich`, `duckdb`, `pandas`, and `numpy`
- Inspired by modular quant research platforms

---

Happy Backtesting! 📈
