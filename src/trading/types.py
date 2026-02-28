from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any

import pandas as pd


LONG = 1
SHORT = -1
FLAT = 0


@dataclass
class Position:
    """回测中的单个持仓对象。"""
    symbol: str
    side: int
    entry_time: pd.Timestamp
    signal_time: pd.Timestamp
    entry_price: float
    qty: float
    leverage: float
    stop_price: float
    take_price: float
    entry_reason: str
    entry_fee: float = 0.0
    entry_slippage_cost: float = 0.0
    funding_pnl: float = 0.0
    holding_bars: int = 0


@dataclass
class Trade:
    """已完成交易记录，用于报表与统计。"""
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    qty: float
    leverage: float
    pnl_usd: float
    pnl_pct: float
    holding_minutes: float
    exit_reason: str
    fees: float
    slippage_cost: float
    funding_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        """将交易对象转成可序列化字典（时间转字符串）。"""
        data = asdict(self)
        data["entry_time"] = self.entry_time.strftime("%Y-%m-%d %H:%M:%S")
        data["exit_time"] = self.exit_time.strftime("%Y-%m-%d %H:%M:%S")
        return data


@dataclass
class Metrics:
    """回测绩效指标集合。"""
    annual_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pct: float = 0.0
    exposure: float = 0.0
    turnover: float = 0.0
    trade_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """将指标对象转换为普通字典。"""
        return asdict(self)


@dataclass
class BacktestResult:
    """单次回测输出结果。"""
    metrics: Metrics
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    strategy_params: Dict[str, Any]
    symbol_execution: Dict[str, pd.DataFrame] = field(default_factory=dict)
