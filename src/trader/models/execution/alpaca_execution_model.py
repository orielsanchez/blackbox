import requests

from trader.core.execution import ExecutionModel


class AlpacaExecutionModel(ExecutionModel):
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Content-Type": "application/json",
            }
        )

    def execute(self, symbol: str, action: str, quantity: int) -> None:
        assert action in {"buy", "sell"}
        url = f"{self.base_url}/v2/orders"
        order = {
            "symbol": symbol,
            "qty": quantity,
            "side": action,
            "type": "market",
            "time_in_force": "gtc",
        }
        resp = self.session.post(url, json=order)
        if resp.status_code != 200:
            raise Exception(f"Order failed: {resp.status_code} - {resp.text}")
