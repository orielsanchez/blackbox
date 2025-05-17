"""
──────────────────────────────────────────────────────────────────────
🧠 Backtest System Architecture — Developer Overview
──────────────────────────────────────────────────────────────────────

This system simulates daily trading for a modular quant strategy using
plug-and-play models and a YAML-defined configuration. The execution
loop for each backtest day is fully deterministic and built for
experimentation and diagnostics.

────────────────────────────────────────────────
📅 Daily Execution Flow (pseudocode)
────────────────────────────────────────────────

for each snapshot in OHLCV data:
    date = snapshot["date"]
    prices = snapshot["prices"]
    features = feature_matrix.loc[date]

    # Step 1: Generate raw alpha signals
    raw_signals = alpha_model.predict(features)

    # Step 2: Apply risk constraints (e.g. position limits, leverage)
    filtered_signals = risk_model.apply(raw_signals)

    # Step 3: Apply transaction cost adjustments
    adjusted_signals = tx_cost_model.adjust(filtered_signals)

    # Step 4: Build target portfolio weights
    capital = execution.portfolio_value
    target = portfolio_model.construct(adjusted_signals, capital)

    # Step 5: Simulate trades vs current portfolio
    current_portfolio = tracker.get_portfolio()
    result = execution_model.execute(target, current_portfolio, prices)

    # Step 6: Update portfolio and account state
    tracker.update(result.trades, result.fill_prices)
    execution.portfolio_value = tracker.compute_portfolio_value(prices)
    execution.current_cash = tracker.current_cash

    # Step 7: Record DailyLog with trades, equity, cash, and feedback
    logs.append(DailyLog(...))

    # Step 8: Update diagnostics (PnL, drawdown)
    pnl = equity - prev_equity
    drawdown = (equity - max_equity) / max_equity

────────────────────────────────────────────────
🏗️ Key Roles & Responsibilities
────────────────────────────────────────────────

- AlphaModel.predict()             → feature vector → signal strength (float per symbol)
- RiskModel.apply()                → prune/filter/scale alpha signals
- TransactionCostModel.adjust()   → penalize or rescore based on fill/cost expectations
- PortfolioModel.construct()      → convert final scores → weights (no knowledge of current portfolio)
- ExecutionModel.execute()        → simulate trade fills (uses current portfolio + prices)
- Tracker.update()                → apply trade results, update cash/equity/positions

────────────────────────────────────────────────
📊 Data Structures in Motion
────────────────────────────────────────────────

- OHLCVSnapshots      → raw market data inputs
- FeatureMatrix       → [date, symbol] MultiIndex of engineered features
- SignalSet           → raw alpha outputs
- PortfolioTarget     → desired weights + capital to deploy
- TradeResult         → executed trades + fill prices
- DailyLog            → final snapshot of trades, cash, equity, feedback

"""
