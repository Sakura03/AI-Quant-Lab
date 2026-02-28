from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import pandas as pd


LONG = 1
SHORT = -1
FLAT = 0


@dataclass
class SignalEvent:
    """Standardized signal emitted by each strategy at signal close time."""

    symbol: str
    timestamp: pd.Timestamp
    strategy: str
    side: int
    strength: float
    stop_atr: float
    take_atr: float
    confidence: float
    reason: str


@dataclass
class PositionState:
    """Live position in portfolio ledger."""

    symbol: str
    side: int
    qty: float
    entry_time: pd.Timestamp
    signal_time: pd.Timestamp
    entry_price: float
    stop_price: float
    take_price: float
    strategy: str
    entry_reason: str
    leverage: float
    holding_bars: int = 0
    entry_fee: float = 0.0
    entry_slippage_cost: float = 0.0
    funding_pnl: float = 0.0


@dataclass
class Trade:
    """Closed trade record for analytics and audit."""

    symbol: str
    strategy: str
    side: str
    entry_time: pd.Timestamp
    signal_time: pd.Timestamp
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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entry_time"] = self.entry_time.strftime("%Y-%m-%d %H:%M:%S")
        data["signal_time"] = self.signal_time.strftime("%Y-%m-%d %H:%M:%S")
        data["exit_time"] = self.exit_time.strftime("%Y-%m-%d %H:%M:%S")
        return data


@dataclass
class Metrics:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    metrics: Metrics
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    strategy_bundle: dict[str, Any]
    symbol_execution: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass
class WindowResult:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    genome: dict[str, Any]
    train_metrics: dict[str, Any]
    val_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    train_score: float
    val_score: float
    test_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.strftime("%Y-%m-%d"),
            "train_end": self.train_end.strftime("%Y-%m-%d"),
            "val_start": self.val_start.strftime("%Y-%m-%d"),
            "val_end": self.val_end.strftime("%Y-%m-%d"),
            "test_start": self.test_start.strftime("%Y-%m-%d"),
            "test_end": self.test_end.strftime("%Y-%m-%d"),
            "train_score": self.train_score,
            "val_score": self.val_score,
            "test_score": self.test_score,
            **{f"train_{k}": v for k, v in self.train_metrics.items()},
            **{f"val_{k}": v for k, v in self.val_metrics.items()},
            **{f"test_{k}": v for k, v in self.test_metrics.items()},
        }


@dataclass
class ExperimentResult:
    best_genome: dict[str, Any]
    stitched_test_metrics: dict[str, Any]
    window_results: list[WindowResult]
    all_trials: pd.DataFrame
    stitched_equity: pd.DataFrame
    stitched_trades: pd.DataFrame
