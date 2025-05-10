## Goal: Live Trading System

1. Ingest current Alpaca data for tracked symbols
2. Use existing models (alpha, risk, portfolio, execution)
3. Log incoming market data to DuckDB
4. Propose trade deltas vs. current positions
5. Wait for user confirmation
6. Execute via Alpaca, track fills + slippage
7. Log performance + metrics for feedback into models
