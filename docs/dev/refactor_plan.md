
# ✅ Full Program Flow (v2 Clean Architecture)

---

## 1. 📄 CLI Entry Point: `main.py`

```python
def main():
    config = load_config("path/to/config.yaml")
    logger = setup_logger(config)

    universe = load_universe(config.universe_file)
    models = build_models(config)
    context = StrategyContext(config, logger, models, PositionTracker())

    # 🔁 Data pipeline
    feature_specs = collect_all_feature_specs(config)
    data = prepare_data_bundle(
        data_config=config.data,
        feature_specs=feature_specs,
        universe=universe,
        run_id=config.run_id,
        force_reload=config.data.force_reload
    )

    # ▶️ Backtest loop
    logs = run_backtest_loop(context, data)

    # 📊 Metrics
    metrics = compute_metrics(
        logs=logs,
        initial_equity=context.initial_equity,
        risk_free_rate=config.risk_free_rate,
        include_equity_curve=config.plot_equity
    )

    # 💾 Output
    write_results(logs, metrics, config, Path("results") / config.run_id)
```

---

## 2. 📦 StrategyContext

Centralizes runtime state (config, logger, models, tracker).

```python
@dataclass
class StrategyContext:
    config: BacktestConfig
    logger: RichLogger
    models: StrategyModels
    tracker: PositionTracker
    initial_equity: float
```

---

## 3. 🔁 Data Pipeline: `prepare_data_bundle(...)`

```python
@dataclass
class PreparedDataBundle:
    snapshots: list[OHLCVSnapshot]
    feature_matrix: pd.DataFrame
    metadata: FeatureMatrixInfo
```

* Combines raw OHLCV and all feature specs
* Computes `max_window` across models
* Loads data via `DuckDBDataLoader`
* Computes feature matrix (or loads from cache)
* Converts to `OHLCVSnapshot`s

---

## 4. 🏃 Simulation: `run_backtest_loop(...)`

```python
def run_backtest_loop(
    ctx: StrategyContext,
    data: PreparedDataBundle
) -> list[DailyLog]:
```

* Iterates over `data.snapshots`
* Slices features using `data.feature_matrix`
* Calls `process_trading_day(...)`
* Updates tracker, appends `DailyLog`

---

## 5. ⚙️ Daily Execution: `process_trading_day(...)`

```python
def process_trading_day(
    snapshot: OHLCVSnapshot,
    features_today: pd.Series,
    features_window: pd.DataFrame,
    ctx: StrategyContext
) -> DailyLog:
```

Steps:

* Alpha → raw scores from `features_today`
* Risk/Cost → applied over `features_window`
* Portfolio → generates weights
* Execution → uses `snapshot.open` for realistic fills
* Tracker → updated
* Log returned with cash, equity, trades, PnL

---

## 6. 📊 Metrics: `compute_metrics(...)`

```python
@dataclass
class BacktestMetrics:
    summary: dict[str, float]
    equity_curve: Optional[pd.Series] = None
```

```python
def compute_metrics(
    logs: list[DailyLog],
    initial_equity: float,
    risk_free_rate: float = 0.0,
    include_equity_curve: bool = False
) -> BacktestMetrics
```

Returns structured summary + optional equity curve.

No more ambiguous tuple.

---

## 7. 💾 Output: `write_results(...)`

```python
def write_results(
    logs: list[DailyLog],
    metrics: BacktestMetrics,
    config: BacktestConfig,
    output_dir: Path
) -> None:
```

Saves:

* History CSV (`DailyLog`)
* JSON summary (`metrics.summary`)
* Plot (if `plot_equity` enabled)
* Equity curve (optional)

---

## 🔐 All Types Are Now:

* ✅ Explicit
* ✅ Typed (no `Any`)
* ✅ Pyright-compliant
* ✅ Decoupled per responsibility

---

## 📌 High-Level Responsibility Map

| Component             | Responsibility                                 |
| --------------------- | ---------------------------------------------- |
| `main.py`             | Orchestration only                             |
| `prepare_data_bundle` | Loads OHLCV, generates/caches features         |
| `run_backtest_loop`   | Calls `process_trading_day` over all snapshots |
| `process_trading_day` | Pure day logic: alpha → risk → trades          |
| `compute_metrics`     | Turns logs into performance summary            |
| `write_results`       | Saves logs, metrics, plot, equity curve        |

---
