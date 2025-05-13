import pandas as pd


class AlpacaMarketExecutionModel:
    name = "alpaca_market"

    def __init__(self, api_key, secret_key, account_id):
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(api_key, secret_key)
        self.account_id = account_id

    def record(self, trades: pd.Series, feedback: dict):
        # Log or store trades locally
        pass

    def update_portfolio(self, current: pd.Series, trades: pd.Series) -> pd.Series:
        # Ideally you query your Alpaca portfolio here
        # For now, simulate locally:
        return current.add(trades, fill_value=0.0)
