from typing import List, Dict, Optional, Any

import os.path as osp
import logging
import pandas as pd

from .logger import BaseClassWithLogger
from .enums import PositionSide, ActionType
from .structs import Balance, Position, ClosedPosition, SymbolData, MarketData, Action
from .indicator import add_indicator
from .performance import PerformanceAnalyzer
from .utils import parse_dataframe, fetch_lastest_data, truncate_dataframe, calculate_liquidation_price
from .visualization import visualize_funding_curve, visualize_candle_and_position, visualize_ichimoku


class BacktestManger(BaseClassWithLogger):
    def __init__(
            self,
            start_time: pd.Timestamp,
            end_time: pd.Timestamp,
            initial_balance: float,
            data_folder: str,
            symbols: List[str],
            indicators: Dict[str, Any],
            analyzer: Optional[PerformanceAnalyzer] = None,
            logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger=logger)

        self.start_time = start_time
        self.end_time = end_time

        self.initial_balance = initial_balance

        self.symbols = symbols
        self.timeframes = list(indicators.keys())
        self.indicators = indicators
        self.analyzer = analyzer

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

            data_path = osp.join(data_folder, f"{symbol_:s}-funding-rate.feather")
            df = parse_dataframe(data_path)
            if df is None:
                raise FileNotFoundError(f"找不到数据: {data_path:s}")
            self.data_dict[symbol]["fundingRate"] = df

    def get_balance(self) -> Balance:
        return self.balance

    def get_position_index(self, symbol: str) -> Optional[int]:
        for i, position in enumerate(self.open_positions):
            if position.symbol == symbol:
                return i
        return None

    def get_positions(self) -> List[Position]:
        return self.open_positions

    def get_closed_positions(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> List[ClosedPosition]:
        return [cp for cp in self.closed_positions if start_time < cp.exit_time <= end_time]

    def get_mark_price(self, symbol: str, current_time: pd.Timestamp) -> Optional[float]:
        mark_price, = fetch_lastest_data(self.data_dict[symbol]["1m"], current_time, cols=["close"])
        return float(mark_price) if mark_price is not None else None

    def get_funding_rate(self, symbol: str, current_time: pd.Timestamp) -> Optional[float]:
        funding_rate, = fetch_lastest_data(self.data_dict[symbol]["fundingRate"], current_time, cols=["fundingRate"], available_until_next_period=False)
        return float(funding_rate) if funding_rate is not None else None

    def get_market_data(self, current_time: pd.Timestamp, limit: int = 1000) -> MarketData:
        return {
            symbol: SymbolData(
                symbol=symbol,
                mark_price=self.get_mark_price(symbol, current_time),
                open_interest=None,
                funding_rate=self.get_funding_rate(symbol, current_time),
                data={
                    timeframe: truncate_dataframe(self.data_dict[symbol][timeframe], current_time, limit=limit)
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
        liquidation_price = calculate_liquidation_price(side == PositionSide.Long, entry_price, action.leverage)
        self.open_positions.append(Position(
            symbol=action.symbol,
            entry_time=current_time,
            side=side,
            entry_price=entry_price,
            mark_price=entry_price,
            quantity=action.position_size_usd / entry_price,
            leverage=action.leverage,
            stop_loss=action.stop_loss,
            take_profit=action.take_profit,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            liquidation_price=liquidation_price,
            margin_used=action.position_size_usd / action.leverage,
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

    def adjust_order(self, index: int, action: Action):
        position = self.open_positions[index]
        position.stop_loss = action.stop_loss
        position.take_profit = action.take_profit

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

        elif action.type == ActionType.AdjustOrder:
            # fetch current position
            index = self.get_position_index(action.symbol)
            if index is None:
                self.warning(f"调整订单失败: {action.symbol}的仓位不存在")
                return

            # adjust order
            self.adjust_order(index, action)

    def execute_actions(self, actions: List[Action], current_time: pd.Timestamp):
        for action in actions:
            self.execute_action(action, current_time)

    def tick(self, current_time: pd.Timestamp):
        # 倒序访问, 防止平仓导致未处理仓位的索引改变
        for i in reversed(range(len(self.open_positions))):
            position = self.open_positions[i]

            position_closed = False
            if position.stop_loss is not None or position.take_profit is not None:
                df = self.data_dict[position.symbol]["1m"]
                start_time = max(self.last_tick, position.entry_time)
                df = truncate_dataframe(df, start_time=start_time, end_time=current_time)

                stop_loss_hit = pd.Series(False, index=df.index)
                take_profit_hit = pd.Series(False, index=df.index)
                if position.stop_loss is not None:
                    if position.side == PositionSide.Long:
                        stop_loss_hit = df["low"] < position.stop_loss
                    else:
                        stop_loss_hit = df["high"] > position.stop_loss
                if position.take_profit is not None:
                    if position.side == PositionSide.Long:
                        take_profit_hit = df["high"] > position.take_profit
                    else:
                        take_profit_hit = df["low"] < position.take_profit

                if stop_loss_hit.any() or take_profit_hit.any():
                    stop_loss_idx = stop_loss_hit.idxmax() if stop_loss_hit.any() else None
                    take_profit_idx = take_profit_hit.idxmax() if take_profit_hit.any() else None

                    # 止损优先
                    if stop_loss_idx is not None and (take_profit_idx is None or stop_loss_idx <= take_profit_idx):
                        exit_idx = stop_loss_idx
                        exit_price = position.stop_loss
                        reason = "止损"
                    else:
                        exit_idx = take_profit_idx
                        exit_price = position.take_profit
                        reason = "止盈"

                    # close position
                    exit_time: pd.Timestamp = df.loc[exit_idx, "timestamp"]
                    self.close_position(i, exit_time, exit_price=exit_price)
                    position_closed = True
                    self.info(f">>> 自动平仓: {position.symbol:s} {position.side.name:s} | 时间: {exit_time.strftime("%Y-%m-%d %H:%M:%S"):s} | 开仓价: {position.entry_price:.4e} | {reason:s}价: {exit_price:.4e}")

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
        if self.analyzer:
            self.info(self.analyzer.get_metrics().format())

    def visualize(self, save_folder: str):
        if self.analyzer:
            visualize_funding_curve(self.analyzer.balance, save_path=osp.join(save_folder, "funding.png"), funding_col="equity")

        position_data = pd.DataFrame(columns=["symbol", "position_side", "buy_time", "buy_price", "sell_time", "sell_price", "quantity", "leverage", "pnl", "pnl_pct"])
        for i, cp in enumerate(self.closed_positions):
            if cp.side == PositionSide.Long:
                position_data.loc[i] = [cp.symbol, cp.side.name, cp.entry_time, cp.entry_price, cp.exit_time, cp.exit_price, cp.quantity, cp.leverage, cp.pnl, cp.pnl_pct]
            else:
                position_data.loc[i] = [cp.symbol, cp.side.name, cp.exit_time, cp.exit_price, cp.entry_time, cp.entry_price, cp.quantity, cp.leverage, cp.pnl, cp.pnl_pct]

        visualize_candle_and_position(self.symbols, self.timeframes, self.start_time, self.end_time, self.data_dict, position_data, save_folder)
        visualize_ichimoku(self.symbols, self.start_time, self.end_time, self.data_dict, position_data, save_folder, indicator_params=self.indicators)
        position_data.to_csv(osp.join(save_folder, "positions.csv"), index=False)
