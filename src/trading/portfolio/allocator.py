from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from trading.domain.types import LONG, PositionState, Trade
from trading.execution.simulator import ExecutionSimulator


class PortfolioLedger:
    """Stateful portfolio ledger for positions, cash, trades, and equity snapshots."""

    def __init__(self, initial_balance: float, execution: ExecutionSimulator):
        """Initialize ledger with starting cash and empty position/trade state."""
        self.initial_balance = float(initial_balance)
        self.cash = float(initial_balance)
        self.execution = execution

        self.positions: dict[str, PositionState] = {}
        self.trades: list[Trade] = []
        self.equity_records: list[dict[str, Any]] = []

        self.turnover_notional = 0.0
        self.peak_equity = float(initial_balance)

    def symbol_notional(self, symbol: str, mark_prices: dict[str, float]) -> float:
        """Compute one symbol's current absolute notional exposure."""
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        px = mark_prices.get(symbol, pos.entry_price)
        return abs(px * pos.qty)

    def gross_notional(self, mark_prices: dict[str, float]) -> float:
        """Compute total absolute notional exposure across open positions."""
        return sum(self.symbol_notional(s, mark_prices) for s in self.positions)

    def unrealized_pnl(self, mark_prices: dict[str, float]) -> float:
        """Compute total unrealized PnL using provided mark prices."""
        pnl = 0.0
        for symbol, pos in self.positions.items():
            px = mark_prices.get(symbol, pos.entry_price)
            pnl += (px - pos.entry_price) * pos.qty * pos.side
        return pnl

    def mark_to_market(self, mark_prices: dict[str, float]) -> float:
        """Mark portfolio to market and update running equity peak."""
        equity = self.cash + self.unrealized_pnl(mark_prices)
        self.peak_equity = max(self.peak_equity, equity)
        return equity

    def current_drawdown(self, mark_prices: dict[str, float]) -> float:
        """Return current drawdown ratio from peak equity."""
        equity = self.mark_to_market(mark_prices)
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - equity / self.peak_equity)

    def open_position(
        self,
        symbol: str,
        side: int,
        timestamp: pd.Timestamp,
        signal_time: pd.Timestamp,
        strategy: str,
        entry_reason: str,
        entry_price: float,
        raw_price: float,
        qty: float,
        leverage: float,
        stop_price: float,
        take_price: float,
    ) -> bool:
        """Open a new position and book entry costs."""
        if symbol in self.positions:
            return False
        if qty <= 0 or entry_price <= 0:
            return False

        entry_fee = self.execution.calc_fee(entry_price, qty)
        entry_slippage_cost = abs(entry_price - raw_price) * qty
        self.cash -= entry_fee
        self.turnover_notional += abs(entry_price * qty)

        self.positions[symbol] = PositionState(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_time=timestamp,
            signal_time=signal_time,
            entry_price=entry_price,
            stop_price=stop_price,
            take_price=take_price,
            strategy=strategy,
            entry_reason=entry_reason,
            leverage=leverage,
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
        raw_price: float,
        reason: str,
    ) -> Trade | None:
        """Close one position, realize PnL, and append one trade record."""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None

        exit_fee = self.execution.calc_fee(exit_price, pos.qty)
        exit_slippage_cost = abs(exit_price - raw_price) * pos.qty
        gross_pnl = (exit_price - pos.entry_price) * pos.qty * pos.side
        net_pnl = gross_pnl - pos.entry_fee - exit_fee + pos.funding_pnl

        self.cash += gross_pnl
        self.cash -= exit_fee
        self.turnover_notional += abs(exit_price * pos.qty)

        holding_minutes = (timestamp - pos.entry_time).total_seconds() / 60.0
        notional = max(abs(pos.entry_price * pos.qty), 1e-12)
        pnl_pct = 100.0 * net_pnl / notional

        trade = Trade(
            symbol=symbol,
            strategy=pos.strategy,
            side="LONG" if pos.side == LONG else "SHORT",
            entry_time=pos.entry_time,
            signal_time=pos.signal_time,
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
        """Apply one funding cashflow to an open position."""
        pos = self.positions.get(symbol)
        if pos is None:
            return
        notional = abs(mark_price * pos.qty)
        funding_pnl = -pos.side * funding_rate * notional
        pos.funding_pnl += funding_pnl
        self.cash += funding_pnl

    def increment_holding_bar(self, symbol: str):
        """Increment holding-bar counter for one open position."""
        pos = self.positions.get(symbol)
        if pos is not None:
            pos.holding_bars += 1

    def snapshot(self, timestamp: pd.Timestamp, mark_prices: dict[str, float]):
        """Append one portfolio snapshot row at current timestamp."""
        equity = self.mark_to_market(mark_prices)
        gross = self.gross_notional(mark_prices)
        drawdown = 0.0 if self.peak_equity <= 0 else max(0.0, 1.0 - equity / self.peak_equity)

        max_symbol_share = 0.0
        if gross > 0:
            max_symbol_share = max(self.symbol_notional(s, mark_prices) / gross for s in self.positions)

        self.equity_records.append(
            {
                "timestamp": timestamp,
                "equity": equity,
                "cash": self.cash,
                "open_positions": len(self.positions),
                "gross_notional": gross,
                "turnover": self.turnover_notional,
                "drawdown": drawdown,
                "max_symbol_share": max_symbol_share,
            }
        )

    def equity_frame(self) -> pd.DataFrame:
        """Build deduplicated equity time series and normalized drawdown column."""
        if not self.equity_records:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "equity",
                    "cash",
                    "open_positions",
                    "gross_notional",
                    "turnover",
                    "drawdown",
                    "max_symbol_share",
                ]
            )
        out = pd.DataFrame(self.equity_records)
        out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        peak = out["equity"].cummax().replace(0.0, pd.NA)
        out["drawdown"] = (1.0 - out["equity"] / peak).fillna(0.0).clip(lower=0.0)
        return out.reset_index(drop=True)

    def trades_frame(self) -> pd.DataFrame:
        """Build trade ledger as DataFrame with stable schema even when empty."""
        if not self.trades:
            cols = list(
                asdict(
                    Trade(
                        symbol="",
                        strategy="",
                        side="",
                        entry_time=pd.Timestamp("1970-01-01"),
                        signal_time=pd.Timestamp("1970-01-01"),
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
                    )
                ).keys()
            )
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([x.to_dict() for x in self.trades])
