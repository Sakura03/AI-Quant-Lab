import numpy as np
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

        nan_to_zero = lambda x: float(0.0 if np.isnan(x) or np.isinf(x) else x)
        return Metrics(
            annual_return=nan_to_zero(emp.annual_return(returns, annualization=self.annualization)),
            return_std=nan_to_zero(np.std(returns, ddof=1)),
            sharpe_ratio=nan_to_zero(emp.sharpe_ratio(returns, annualization=self.annualization)),
            sortino_ratio=nan_to_zero(emp.sortino_ratio(returns, annualization=self.annualization)),
            calmar_ratio=nan_to_zero(emp.calmar_ratio(returns, annualization=self.annualization)),
            max_drawdown=nan_to_zero(emp.max_drawdown(returns)),
        )
