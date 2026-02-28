from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any

import pandas as pd

from .execution import ExecutionModel
from .types import Position, Trade, LONG


class Portfolio:
    """资金与持仓账本，负责交易记账和权益快照。"""
    def __init__(self, initial_balance: float, execution_model: ExecutionModel):
        """初始化组合状态。"""
        self.initial_balance = float(initial_balance)
        self.cash = float(initial_balance)
        self.execution_model = execution_model

        self.positions: Dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_records: list[Dict[str, Any]] = []

        self.turnover_notional = 0.0

    def gross_notional(self, mark_prices: Dict[str, float]) -> float:
        """计算当前所有持仓的总名义价值（绝对值求和）。"""
        gross = 0.0
        for symbol, pos in self.positions.items():
            px = mark_prices.get(symbol)
            if px is None:
                px = pos.entry_price
            gross += abs(px * pos.qty)
        return gross

    def unrealized_pnl(self, mark_prices: Dict[str, float]) -> float:
        """计算当前所有持仓的未实现盈亏（按标记价格计价）。"""
        pnl = 0.0
        for symbol, pos in self.positions.items():
            px = mark_prices.get(symbol)
            if px is None:
                px = pos.entry_price
            pnl += (px - pos.entry_price) * pos.qty * pos.side
        return pnl

    def mark_to_market(self, mark_prices: Dict[str, float]) -> float:
        """按市价计算账户权益（现金 + 未实现盈亏）。"""
        return self.cash + self.unrealized_pnl(mark_prices)

    def open_position(
        self,
        symbol: str,
        side: int,
        timestamp: pd.Timestamp,
        signal_time: pd.Timestamp,
        entry_price: float,
        qty: float,
        leverage: float,
        stop_price: float,
        take_price: float,
        reason: str,
        raw_price: float,
    ) -> bool:
        """开仓并写入持仓账本，返回是否开仓成功。"""
        if symbol in self.positions:
            return False
        if qty <= 0 or entry_price <= 0:
            return False

        entry_fee = self.execution_model.calc_fee(entry_price, qty)
        entry_slippage_cost = abs(entry_price - raw_price) * qty
        self.cash -= entry_fee
        self.turnover_notional += abs(entry_price * qty)

        self.positions[symbol] = Position(
            symbol=symbol,
            side=side,
            entry_time=timestamp,
            signal_time=signal_time,
            entry_price=entry_price,
            qty=qty,
            leverage=leverage,
            stop_price=stop_price,
            take_price=take_price,
            entry_reason=reason,
            entry_fee=entry_fee,
            entry_slippage_cost=entry_slippage_cost,
            funding_pnl=0.0,
            holding_bars=0,
        )
        return True

    def close_position(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        exit_price: float,
        reason: str,
        raw_price: float,
    ) -> Trade | None:
        """平仓并结算交易成本，返回成交后的交易记录。"""
        if symbol not in self.positions:
            return None

        pos = self.positions.pop(symbol)

        exit_fee = self.execution_model.calc_fee(exit_price, pos.qty)
        exit_slippage_cost = abs(exit_price - raw_price) * pos.qty
        gross_pnl = (exit_price - pos.entry_price) * pos.qty * pos.side
        net_pnl = gross_pnl - pos.entry_fee - exit_fee + pos.funding_pnl

        self.cash += gross_pnl
        self.cash -= exit_fee
        self.turnover_notional += abs(exit_price * pos.qty)

        holding_minutes = (timestamp - pos.entry_time).total_seconds() / 60.0
        notional = max(abs(pos.entry_price * pos.qty), 1e-12)
        pnl_pct = net_pnl / notional * 100.0

        trade = Trade(
            symbol=symbol,
            side="LONG" if pos.side == LONG else "SHORT",
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=timestamp,
            exit_price=exit_price,
            qty=pos.qty,
            leverage=pos.leverage,
            pnl_usd=net_pnl,
            pnl_pct=pnl_pct,
            holding_minutes=holding_minutes,
            exit_reason=reason,
            fees=pos.entry_fee + exit_fee,
            slippage_cost=pos.entry_slippage_cost + exit_slippage_cost,
            funding_pnl=pos.funding_pnl,
        )
        self.trades.append(trade)
        return trade

    def apply_funding(self, symbol: str, funding_rate: float, mark_price: float):
        """对指定持仓结算一次资金费（直接计入现金）。"""
        pos = self.positions.get(symbol)
        if pos is None:
            return

        notional = abs(mark_price * pos.qty)
        funding_pnl = -pos.side * funding_rate * notional
        pos.funding_pnl += funding_pnl
        self.cash += funding_pnl

    def increment_holding_bar(self, symbol: str):
        """将指定持仓的持仓 bar 计数加一。"""
        pos = self.positions.get(symbol)
        if pos is not None:
            pos.holding_bars += 1

    def snapshot(self, timestamp: pd.Timestamp, mark_prices: Dict[str, float]):
        """记录某时刻的权益快照，用于后续绩效分析。"""
        equity = self.mark_to_market(mark_prices)
        self.equity_records.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "cash": self.cash,
                "open_positions": len(self.positions),
                "gross_notional": self.gross_notional(mark_prices),
                "turnover": self.turnover_notional,
            }
        )

    def equity_frame(self) -> pd.DataFrame:
        """返回按时间排序的权益曲线 DataFrame。"""
        if not self.equity_records:
            return pd.DataFrame(columns=["timestamp", "equity", "cash", "open_positions", "gross_notional", "turnover"])
        df = pd.DataFrame(self.equity_records)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        return df.reset_index(drop=True)

    def trades_frame(self) -> pd.DataFrame:
        """返回交易记录 DataFrame；无交易时返回带列名的空表。"""
        if not self.trades:
            cols = list(asdict(Trade(
                symbol="",
                side="",
                entry_time=pd.Timestamp("1970-01-01"),
                entry_price=0.0,
                exit_time=pd.Timestamp("1970-01-01"),
                exit_price=0.0,
                qty=0.0,
                leverage=1.0,
                pnl_usd=0.0,
                pnl_pct=0.0,
                holding_minutes=0.0,
                exit_reason="",
                fees=0.0,
                slippage_cost=0.0,
                funding_pnl=0.0,
            )).keys())
            return pd.DataFrame(columns=cols)

        return pd.DataFrame([t.to_dict() for t in self.trades])
