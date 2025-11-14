import pandas as pd
import empyrical as emp

from .structs import Balance, Metrics
from .utils import timeframe_to_seconds


class PerformanceAnalyzer:
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.annualization = int(86400 * 365 / timeframe_to_seconds(timeframe))

        self.balance = pd.DataFrame(columns=["equity", "available"])
        self.balance.index.name = "timestamp"   # 设置 index 名称

    def add_balance(self, timestamp: pd.Timestamp, balance: Balance):
        self.balance.loc[timestamp] = [balance.total_wallet_balance + balance.total_unrealized_profit, balance.available_balance]

    def get_metrics(self) -> Metrics:
        if len(self.balance) < 10:
            # 十个周期内不计算夏普比率等
            return Metrics()

        # 计算每日收益率
        returns = self.balance["equity"].pct_change().dropna()

        return Metrics(
            annual_return=float(emp.annual_return(returns, annualization=self.annualization)),
            sharpe_ratio=float(emp.sharpe_ratio(returns, annualization=self.annualization)),
            sortino_ratio=float(emp.sortino_ratio(returns, annualization=self.annualization)),
            calmar_ratio=float(emp.calmar_ratio(returns, annualization=self.annualization)),
            max_drawdown=float(emp.max_drawdown(returns)),
        )
