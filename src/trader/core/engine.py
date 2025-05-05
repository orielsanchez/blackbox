from dataclasses import dataclass

from trader.core.alpha import AlphaModel
from trader.core.execution import ExecutionModel
from trader.core.portfolio import PortfolioConstructionModel
from trader.core.risk import RiskModel
from trader.core.slippage import SlippageModel
from trader.core.tx_cost import TransactionCostModel


@dataclass
class ModelBundle:
    alpha: AlphaModel
    risk: RiskModel
    tx_cost: TransactionCostModel
    portfolio: PortfolioConstructionModel
    execution: ExecutionModel
    slippage: SlippageModel
