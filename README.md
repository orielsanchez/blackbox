# BlackBox Backtesting Framework

A modular, extensible backtesting engine for quantitative trading strategies. Built for research and production-aligned simulation. Components are plug-and-play and configured via YAML.

## 📦 Features

- **Modular architecture**: alpha, risk, cost, portfolio, execution models
- **YAML-driven strategy configuration**: define strategies without coding
- **DuckDB-based historical data backend**: efficient storage and retrieval
- **Comprehensive simulation**: slippage, commission, transaction costs
- **Robust risk controls**: leverage, position sizing, short constraints
- **Detailed diagnostics**: logging at each step of the trading process
- **Performance analytics**: Sharpe, max drawdown, annualized return
- **Extensible design**: registries and model inheritance for customization
- **MultiIndex support**: proper handling of complex data structures

## 📈 Latest Updates

- **Fixed volatility calculation**: Resolved critical bug in portfolio construction
- **Enhanced error handling**: Added robust fallbacks for calculation edge cases
- **Improved diagnostic logging**: Better tracking of transformations at each step
- **Better parameter flexibility**: Fine-grained control of risk and position sizing

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

# Single source of truth for portfolio value
initial_portfolio_value: 100000

data:
  db_path: db/ohlcv.duckdb
  rolling: true
  window: 5

alpha_model:
  name: mean_reversion
  params:
    window: 5
    threshold: 0.5
    features:
      - name: zscore_price
        params:
          period: 20
      - name: bollinger_band
        params:
          period: 20
          std_dev: 2.0

risk_model:
  name: position_limit
  params:
    max_leverage: 1.0
    max_position_size: 0.25  # Limit concentration risk
    allow_shorts: true

tx_cost_model:
  name: quadratic_market_impact
  params:
    commission_rate: 0.0001
    impact_coefficient: 5e-5
    min_commission: 1.0

portfolio_model:
  name: volatility_scaled
  params:
    vol_lookback: 20
    risk_target: 0.5  # 50% of capital as risk budget
    max_weight: 0.25  # Limit per-position concentration
    min_notional: 10.0
    min_price: 1.0

execution_model:
  name: market
  params:
    slippage: 0.0001
    commission: 0.0001
    fractional: true
    allow_shorts: true
    min_notional: 10.0

min_holding_period: 1
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
  - Sortino Ratio
  - Max Drawdown (%)
  - Calmar Ratio

Output saved in `results/<run_id>/`.

## 📁 Data Format

OHLCV data is loaded via DuckDB from a table with structure:
```
symbol | date | open | high | low | close | volume
```

Universe files are simple CSVs:
```
symbol
AAPL
GOOG
MSFT
```

## 🔧 Extending

Add new models by subclassing the appropriate interface and registering:

```python
@register_model("my_alpha")
class MyAlpha(AlphaModel):
    name = "my_alpha"
    
    def __init__(self, param1=0.5, param2=20, features=None):
        # Initialize parameters
        self.param1 = param1
        self.param2 = param2
        super().__init__(features)
        
    def generate(self, snapshot: dict) -> pd.Series:
        # Generate alpha signals
        # Return pd.Series [symbol → signal]
```

Models are auto-discovered and loaded based on the config file.

## 🎯 Troubleshooting

- **No trades executing**: Check for overlap between alpha symbols and volatility calculation
- **NaN values in normalization**: Ensure signal ranges are non-uniform and contain valid data
- **Index mismatches**: Verify OHLCV data has proper MultiIndex structure ['date', 'symbol']
- **Performance issues**: Enable DEBUG logging to trace signal transformations

## 📝 License

MIT License

## ✨ Credits

- Built with `rich`, `duckdb`, `pandas`, and `numpy`
- Inspired by modular quant research platforms

---

Happy Backtesting! 📈
