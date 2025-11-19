from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

import pandas as pd

from .enums import PositionSide, ActionType
from .utils import timeframe_to_seconds, format_time_interval, format_symbol, infer_period


@dataclass
class Balance:
    total_wallet_balance: float
    total_unrealized_profit: float
    available_balance: float

    @property
    def available_balance_pct(self) -> float:
        if self.total_wallet_balance == 0.0:
            return 0.0
        return self.available_balance / self.total_wallet_balance * 100

    @property
    def total_unrealized_profit_pct(self) -> float:
        if self.total_wallet_balance == 0.0:
            return 0.0
        return self.total_unrealized_profit / self.total_wallet_balance * 100

    @property
    def margin_used(self) -> float:
        return self.total_wallet_balance - self.available_balance

    @property
    def margin_used_pct(self) -> float:
        if self.total_wallet_balance == 0.0:
            return 0.0
        return self.margin_used / self.total_wallet_balance * 100

    def format(self, initial_balance: Optional[float] = None) -> str:
        parts = [
            f"钱包余额: {self.total_wallet_balance:.2f} USDT",
            f"可用保证金: {self.available_balance:.2f} USDT ({self.available_balance_pct:.1f}%)",
            f"已用保证金: {self.margin_used:.2f} USDT ({self.margin_used_pct:.1f}%)",
            f"未实现盈亏: {self.total_unrealized_profit:.2f} USDT ({self.total_unrealized_profit_pct:+.1f}%)",
        ]

        if initial_balance is not None:
            pnl = self.total_wallet_balance - initial_balance
            pnl_pct = pnl / initial_balance * 100
            parts.append(f"已实现盈亏: {pnl:.2f} USDT ({pnl_pct:+.1f}%)")

        return " | ".join(parts)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Balance:
        info = data["info"]
        return Balance(
            total_wallet_balance=float(info["totalWalletBalance"]),
            total_unrealized_profit=float(info["totalUnrealizedProfit"]),
            available_balance=float(info["availableBalance"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    symbol: str
    side: PositionSide
    entry_time: Optional[pd.Timestamp]
    entry_price: float
    mark_price: float
    quantity: float
    leverage: int
    unrealized_pnl: float
    unrealized_pnl_pct:float
    liquidation_price: Optional[float]
    margin_used: float

    meta: Dict[str, Any] = field(default_factory=dict)

    def format(self, current_time: pd.Timestamp) -> str:
        parts = [
            f"{self.symbol:s} {self.side.name:s}",
            f"入场价: {self.entry_price:.4e}",
            f"当前价: {self.mark_price:.4e}",
            f"当前盈亏: {self.unrealized_pnl:.2f} USDT ({self.unrealized_pnl_pct:+.1f}%)",
            f"杠杆: {self.leverage:d}x",
            f"保证金: {self.margin_used:.2f} USDT",
        ]

        if self.liquidation_price is not None:
            parts.append(f"强平价: {self.liquidation_price:.4e}")

        if self.entry_time:
            holding_seconds = int((current_time - self.entry_time).total_seconds())
            parts.append("持仓时间: " + format_time_interval(holding_seconds))

        return " | ".join(parts)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Position:
        update_time = data.get("info", {}).get("updateTime", None)
        side_str = str(data["side"]).lower()
        return Position(
            symbol=format_symbol(str(data["symbol"])),
            side=PositionSide.Long if side_str in ["long", "buy"] else PositionSide.Short,
            entry_time=pd.to_datetime(int(update_time), unit="ms") if update_time is not None else None,
            entry_price=float(data["entryPrice"]),
            mark_price=float(data["markPrice"]),
            quantity=float(abs(data["contracts"] * data["contractSize"])),
            leverage=int(data["leverage"]),
            unrealized_pnl=float(data["unrealizedPnl"]),
            unrealized_pnl_pct=float(data["percentage"]),
            liquidation_price=float(data["liquidationPrice"]) if data["liquidationPrice"] is not None else None,
            margin_used=float(data["initialMargin"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        ret = asdict(self)

        # 转成可序列化的格式
        ret["side"] = self.side.name
        if self.entry_time:
            ret["entry_time"] = self.entry_time.strftime("%Y-%m-%d %H:%M:%S")

        return ret


@dataclass
class ClosedPosition:
    symbol: str
    side: PositionSide
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int

    @property
    def pnl(self) -> float:
        if self.side == PositionSide.Long:
            return (self.exit_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.Short:
            return (self.entry_price - self.exit_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        if self.side == PositionSide.Long:
            return (self.exit_price / self.entry_price - 1.0) * self.leverage * 100
        else:
            return (self.entry_price / self.exit_price - 1.0) * self.leverage * 100

    def format(self) -> str:
        entry_time_str = self.entry_time.strftime("%Y-%m-%d %H:%M:%S")
        exit_time_str = self.exit_time.strftime("%Y-%m-%d %H:%M:%S")
        holding_time_str = format_time_interval(int((self.exit_time - self.entry_time).total_seconds()))
        parts = [
            f"{self.symbol:s} {self.side.name:s}",
            f"入场时间: {entry_time_str:s}",
            f"入场价: {self.entry_price:.4e}",
            f"离场时间: {exit_time_str:s}",
            f"离场价: {self.exit_price:.4e}",
            f"数量: {self.quantity:.2e}",
            f"杠杆: {self.leverage:d}x",
            f"盈亏: {self.pnl:.2f} USDT ({self.pnl_pct:+.1f}%)",
            f"持仓时间: {holding_time_str:s}",
        ]

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        ret = asdict(self)

        # 转成可序列化的格式
        ret["side"] = self.side.name
        ret["entry_time"] = self.entry_time.strftime("%Y-%m-%d %H:%M:%S")
        ret["exit_time"] = self.exit_time.strftime("%Y-%m-%d %H:%M:%S")

        return ret


@dataclass
class SymbolData:
    symbol: str

    mark_price: float
    open_interest: Optional[float] = None
    funding_rate: Optional[float] = None

    data: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def calculate_pct_change(self, timeframe: str) -> Optional[float]:
        target_interval = timeframe_to_seconds(timeframe)
        for tf in self.data.keys():
            interval = timeframe_to_seconds(tf)
            if target_interval % interval != 0:
                continue

            span = target_interval // interval
            data = self.data[tf]
            if len(data) < span + 1:
                continue

            close = data["close"]
            return (close.iloc[-1] / close.iloc[-span-1] - 1.0) * 100.0

        return None

    def pct_change_str(self, timeframes: List[str] = ["1h", "4h", "1d"]) -> str:
        results = []
        for tf in timeframes:
            pct_change = self.calculate_pct_change(tf)
            if pct_change is not None:
                results.append(f"{tf:s}: {pct_change:+.2f}%")

        if len(results) == 0:
            return ""

        return "(" + ", ".join(results) + ")"

    def format(self, span: int = 50):
        parts = [f"当前标记价格: {self.mark_price:.4e}"]
        if self.open_interest is not None:
            parts.append(f"当前未平仓合约总量: {self.open_interest:.4f}")
        if self.funding_rate is not None:
            parts.append(f"当前资金费率: {self.funding_rate:.2e}")
        text = " | ".join(parts)

        parts = [text]
        for timeframe, df in self.data.items():
            df = df.tail(span).copy()
            period = infer_period(df)
            start_time_str = df["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = (df["timestamp"].iloc[-1] + period).strftime("%Y-%m-%d %H:%M:%S")
            df = df.drop(columns=["timestamp", "open"])
            parts.append(f"时间周期: {timeframe}, 时间范围: {start_time_str:s} 到 {end_time_str:s} (从上到下)\n" + df.to_string(index=False))

        return "\n\n".join(parts)


MarketData = Dict[str, SymbolData]


@dataclass
class Context:
    current_time: pd.Timestamp
    running_time: pd.Timedelta
    num_cycle: int

    balance: Balance
    positions: List[Position]
    market_data: MarketData

    performance: Metrics


@dataclass
class Action:
    symbol: str
    type: ActionType
    leverage: int = 0
    position_size_usd: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence: int = 0
    reasoning: str = ""

    def format(self) -> str:
        parts = [f"{self.symbol:s} {self.type.name:s}"]

        if self.type in [ActionType.OpenLong, ActionType.OpenShort]:
            parts.extend([
                f"仓位: {self.position_size_usd:.2f} USDT",
                f"杠杆: {self.leverage:d}x",
                f"止损价: {self.stop_loss:.4e}",
                f"止盈价: {self.take_profit:.4e}",
            ])

        parts.extend([f"置信度: {self.confidence:d}", f"原因: {self.reasoning:s}"])

        return " | ".join(parts)
                   

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Optional[Action]:
        if not ("symbol" in data and "action" in data):
            return None

        symbol = format_symbol(str(data["symbol"]))
        if data["action"] == "do_nothing":
            action_type = ActionType.DoNothing
        elif data["action"] == "open_long":
            action_type = ActionType.OpenLong
        elif data["action"] == "open_short":
            action_type = ActionType.OpenShort
        elif data["action"] == "close_long":
            action_type = ActionType.CloseLong
        elif data["action"] == "close_short":
            action_type = ActionType.CloseShort
        else:
            return None

        if action_type in [ActionType.DoNothing, ActionType.CloseLong, ActionType.CloseShort]:
            reasoning = str(data["reasoning"]) if "reasoning" in data else ""
            return Action(symbol=symbol, type=action_type, reasoning=reasoning)

        if not ("leverage" in data and "position_size_usd" in data and "stop_loss" in data and "take_profit" in data and "confidence" in data and "reasoning" in data):
            return None

        return Action(
            symbol=symbol,
            type=action_type,
            leverage=int(data["leverage"]),
            position_size_usd=float(data["position_size_usd"]),
            stop_loss=float(data["stop_loss"]),
            take_profit=float(data["take_profit"]),
            confidence=int(data["confidence"]),
            reasoning=str(data["reasoning"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        ret = asdict(self)

        # 转成可序列化的格式
        ret["type"] = self.type.name

        return ret


@dataclass
class Metrics:
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0

    def format(self) -> str:
        parts = [
            f"年化收益率: {self.annual_return*100:.2f}%",
            f"夏普比率: {self.sharpe_ratio:.2f}",
            f"索提诺比率: {self.sortino_ratio:.2f}",
            f"卡玛比率: {self.calmar_ratio:.2f}",
            f"最大回撤: {self.max_drawdown*100:.2f}%",
        ]

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
