from typing import List, Dict, Optional, Any

import os.path as osp
import logging
import pandas as pd

from .logger import BaseClassWithLogger
from .enums import PositionSide, ActionType
from .structs import Balance, Position, ClosedPosition, SymbolData, MarketData, Action
from .indicator import add_indicator
from .utils import parse_dataframe, fetch_lastest_data, truncate_dataframe, calculate_liquidation_price


class BacktestManger(BaseClassWithLogger):
    def __init__(self, start_time: pd.Timestamp, end_time: pd.Timestamp, initial_balance: pd.Timestamp, data_folder: str, symbols: List[str], indicators: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)

        self.start_time = start_time
        self.end_time = end_time

        self.initial_balance = initial_balance

        self.symbols = symbols
        self.timeframes = list(indicators.keys())
        self.indicators = indicators

        self.last_tick = start_time
        self.balance = Balance(total_wallet_balance=initial_balance, total_unrealized_profit=0.0, available_balance=initial_balance)
        self.open_positions: List[Position] = []
        self.closed_positions: List[ClosedPosition] = []

        self.data_dict = { symbol: {} for symbol in symbols }
        for symbol in symbols:
            symbol_ = symbol.replace("/", "_")
            for timeframe, params in indicators.items():
                data_path = osp.join(data_folder, f"{symbol_:s}-{timeframe:s}.feather")
                df = parse_dataframe(data_path)
                if df is None:
                    raise FileNotFoundError(f"找不到数据: {data_path:s}")

                for param in params:
                    add_indicator(df, param)

                self.data_dict[symbol][timeframe] = df

            if "1m" not in self.data_dict[symbol]:
                data_path = osp.join(data_folder, f"{symbol_:s}-1m.feather")
                df = parse_dataframe(data_path)
                if df is None:
                    raise FileNotFoundError(f"找不到数据: {data_path:s}")
                self.data_dict[symbol]["1m"] = df

    def get_balance(self) -> Balance:
        return self.balance

    def get_position_index(self, symbol: str) -> Optional[int]:
        for i, position in enumerate(self.open_positions):
            if position.symbol == symbol:
                return i
        return None

    def get_positions(self) -> List[Position]:
        return self.open_positions

    def get_mark_price(self, symbol: str, current_time: pd.Timestamp) -> Optional[float]:
        mark_price, = fetch_lastest_data(self.data_dict[symbol]["1m"], current_time, cols=["close"])
        return float(mark_price) if mark_price is not None else None

    def get_market_data(self, current_time: pd.Timestamp, limit: int = 1000) -> MarketData:
        return {
            symbol: SymbolData(
                symbol=symbol,
                mark_price=self.get_mark_price(symbol, current_time),
                open_interest=None,
                funding_rate=None,
                data={
                    timeframe: truncate_dataframe(self.data_dict[symbol][timeframe], current_time, limit)
                    for timeframe in self.timeframes
                },
            )
            for symbol in self.symbols
        }

    def open_position(self, action: Action, current_time: pd.Timestamp, entry_price: Optional[float] = None):
        # fetch current price
        if entry_price is None:
            entry_price = self.get_mark_price(action.symbol, current_time)
            if entry_price is None:
                current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                self.warning(f"开仓失败: 无法获得{action.symbol:s}在{current_time_str:s}时刻的最新价格")
                return

        side = PositionSide.Long if action.type == ActionType.OpenLong else PositionSide.Short
        liquidation_price = calculate_liquidation_price(side, entry_price, action.leverage)
        self.open_positions.append(Position(
            symbol=action.symbol,
            entry_time=current_time,
            side=side,
            entry_price=entry_price,
            mark_price=entry_price,
            quantity=action.position_size_usd / entry_price,
            leverage=action.leverage,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            liquidation_price=liquidation_price,
            margin_used=action.position_size_usd / action.leverage,
            meta={
                "stop_loss": action.stop_loss,
                "take_profit": action.take_profit,
            },
        ))

        # update balance
        self.balance.available_balance -= action.position_size_usd / action.leverage

    def close_position(self, index: int, current_time: pd.Timestamp, exit_price: Optional[float] = None):
        if not 0 <= index < len(self.open_positions):
            self.warning(f"索引{index:d}超出边界, 当前持仓数: {len(self.open_positions):d}")
            return

        # fetch current price
        if exit_price is None:
            symbol = self.open_positions[index].symbol
            exit_price = self.get_mark_price(symbol, current_time)
            if exit_price is None:
                current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                self.warning(f"平仓失败: 无法获得{symbol:s}在{current_time_str:s}时刻的最新价格")
                return

        self.update_position(index, exit_price)

        # close position
        position = self.open_positions.pop(index)
        self.closed_positions.append(ClosedPosition(
            symbol=position.symbol,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=current_time,
            entry_price=position.entry_price,
            exit_price=position.mark_price,
            quantity=position.quantity,
            leverage=position.leverage,
        ))

        # update balance
        self.balance.total_wallet_balance += position.unrealized_pnl
        self.balance.total_unrealized_profit -= position.unrealized_pnl
        self.balance.available_balance += position.margin_used + position.unrealized_pnl

    def update_position(self, index: int, current_price: float):
        if not 0 <= index < len(self.open_positions):
            self.warning(f"索引{index:d}超出边界, 当前持仓数: {len(self.open_positions):d}")
            return

        position = self.open_positions[index]
        old_pnl = position.unrealized_pnl

        position.mark_price = current_price
        if position.side == PositionSide.Long:
            position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            position.unrealized_pnl_pct = (current_price / position.entry_price - 1.0) * position.leverage * 100
        elif position.side == PositionSide.Short:
            position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            position.unrealized_pnl_pct = (position.entry_price / current_price - 1.0) * position.leverage * 100

        # update balance
        self.balance.total_unrealized_profit += position.unrealized_pnl - old_pnl

    def execute_action(self, action: Action, current_time: pd.Timestamp):
        if action.type in [ActionType.OpenLong, ActionType.OpenShort]:
            # fetch current position
            index = self.get_position_index(action.symbol)
            if index is not None:
                self.warning(f"开仓失败: {action.symbol}的仓位已存在")
                return

            # open position
            self.open_position(action, current_time)

        elif action.type in [ActionType.CloseLong, ActionType.CloseShort]:
            # fetch current position
            index = self.get_position_index(action.symbol)
            if index is None:
                self.warning(f"平仓失败: {action.symbol}的仓位不存在")
                return

            # close position
            self.close_position(index, current_time)

            # make sure no multiple positions
            assert self.get_position_index(action.symbol) is None

    def execute_actions(self, actions: List[Action], current_time: pd.Timestamp):
        for action in actions:
            self.execute_action(action, current_time)

    def tick(self, current_time: pd.Timestamp):
        # 倒序访问, 防止平仓导致未处理仓位的索引改变
        for i in reversed(range(len(self.open_positions))):
            position = self.open_positions[i]
            stop_loss = position.meta["stop_loss"] if "stop_loss" in position.meta else None
            take_profit = position.meta["take_profit"] if "take_profit" in position.meta else None

            position_closed = False
            if stop_loss is not None or take_profit is not None:
                df = self.data_dict[position.symbol]["1m"]
                start_time = max(self.last_tick, position.entry_time)
                df = df[(df["timestamp"] > start_time) & (df["timestamp"] <= current_time)]
                for j in range(len(df)):
                    ts, high, low = df.iloc[j][["timestamp", "high", "low"]]
                    high, low = float(high), float(low)

                    exit_price, reason = None, None
                    # Consider the worst case: stop loss occurs ahead of take profit
                    if (
                        stop_loss is not None and \
                        (
                            (position.side == PositionSide.Long and low < stop_loss) or \
                            (position.side == PositionSide.Short and high > stop_loss)
                        )
                    ):
                        exit_price = position.meta["stop_loss"]
                        reason = "止损"
                    elif (
                        take_profit is not None and \
                        (
                            (position.side == PositionSide.Long and high > take_profit) or \
                            (position.side == PositionSide.Short and low < take_profit)
                        )
                    ):
                        exit_price = position.meta["take_profit"]
                        reason = "止盈"

                    # close position
                    if exit_price is not None:
                        self.close_position(i, ts, exit_price=exit_price)
                        position_closed = True
                        self.info(f">>> 自动平仓: {position.symbol:s} {position.side.name:s} | 开仓价: {position.entry_price:.4e} | {reason:s}价: {exit_price:.4e}")
                        break

            if not position_closed:
                mark_price = self.get_mark_price(position.symbol, current_time)
                if mark_price is None:
                    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                    self.warning(f"更新持仓信息失败: 无法获得{position.symbol:s}在{current_time_str:s}时刻的最新价格")
                    continue

                self.update_position(i, mark_price)

        self.last_tick = current_time

    def finish(self):
        for i in reversed(range(len(self.open_positions))):
            # close position
            self.close_position(i, self.end_time)
        
        assert len(self.open_positions) == 0

    def analyze(self):
        if len(self.open_positions) > 0:
            self.warning("当前仍有持仓, 请先运行 finish()")
            return

        pnl_pct = (self.balance.total_wallet_balance / self.initial_balance - 1.0) * 100
        self.info(f"初始金额: {self.initial_balance:.2f} USDT, 结束金额: {self.balance.total_wallet_balance:.2f} USDT ({pnl_pct:+.1f}%)")
