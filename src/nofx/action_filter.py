from typing import Tuple, List, Dict, Optional

import logging

from .logger import BaseClassWithLogger
from .enums import PositionSide, ActionType
from .structs import Action, Context


class ActionFilter(BaseClassWithLogger):
    priority_map = {
        ActionType.AdjustOrder: 0,
        ActionType.CloseLong: 1,
        ActionType.CloseShort: 1,
        ActionType.OpenLong: 2,
        ActionType.OpenShort: 2,
        ActionType.DoNothing: 3,
    }

    def __init__(self, altcoin_leverage: int, BTC_ETH_leverage: int, max_positions: int, restricts: Dict[str, Tuple[float, float]], logger: Optional[logging.Logger] = None, **kwargs):
        super().__init__(logger=logger)

        self.altcoin_leverage = altcoin_leverage
        self.BTC_ETH_leverage = BTC_ETH_leverage
        self.max_positions = max_positions

        self.restricts = restricts

    def __call__(self, actions: List[Action], ctx: Context) -> List[Action]:
        actions = self.sort_actions(actions)
        available_balance = ctx.balance.available_balance
        num_position = len(ctx.positions)

        results = []
        position_map = {p.symbol: p for p in ctx.positions}
        for action in actions:
            reason = self.get_invalid_reason(action, ctx, available_balance, num_position)
            if len(reason) == 0:
                results.append(action)
                if action.type in [ActionType.OpenLong, ActionType.OpenShort]:
                    num_position += 1
                    available_balance -= action.position_size_usd / action.leverage
                elif action.type in [ActionType.CloseLong, ActionType.CloseShort]:
                    num_position -= 1

                    open_position = position_map.get(action.symbol, None)
                    if open_position:
                        available_balance += (open_position.margin_used + open_position.unrealized_pnl)
                    else:
                        self.warning(f"ActionFilter失效: {action.symbol:s}的仓位不存在")
            else:
                self.warning(f"无效的决策: {action.format():s} | 无效原因: {reason:s}")

        return results

    def sort_actions(self, actions: List[Action]) -> List[Action]:
        return sorted(actions, key=lambda a: (self.priority_map[a.type], a.position_size_usd / max(1, a.leverage)))

    def get_invalid_reason(self, action: Action, ctx: Context, available_balance: float, num_position: float) -> str:
        if action.symbol not in ctx.market_data.keys():
            return f"未知币种: {action.symbol:s}"

        if action.type in [ActionType.OpenLong, ActionType.OpenShort]:
            if any(action.symbol == position.symbol for position in ctx.positions):
                return f"{action.symbol:s}的仓位已存在"

            if num_position >= self.max_positions:
                return f"持仓数达到最大值: {num_position:d}"

            max_leverage = self.BTC_ETH_leverage if action.symbol in ["BTC/USDT", "ETH/USDT"] else self.altcoin_leverage
            if action.leverage <= 0 or action.leverage > max_leverage:
                return f"杠杆倍数 ({action.leverage:d}x) 大于最大值 ({max_leverage:d}x)"

            if action.position_size_usd < 0.0:
                return f"负仓位: ({action.position_size_usd:.2f} USDT)"

            mark_price = ctx.market_data[action.symbol].mark_price
            if action.symbol in self.restricts:
                min_amount, min_notional = self.restricts[action.symbol]
                amount = action.position_size_usd / mark_price
                if amount < min_amount:
                    return f"{action.symbol:s}的仓位 ({amount:.4f}) 小于最小仓位 ({min_amount:.4f})"
                real_amount = int(amount / min_amount) * min_amount
                notional = real_amount * mark_price
                if notional < min_notional:
                    return f"{action.symbol:s}的名义价值 ({notional:.2f} USDT) 小于最小名义价值 ({min_notional:.2f} USDT)"

            if available_balance < action.position_size_usd / action.leverage:
                return f"保证金不足 (可用保证金: {available_balance:.2f} USDT, 仓位: {action.position_size_usd:.2f} USDT, 杠杆: {action.leverage:d}x)"

            if (
                action.stop_loss < 0.0 or \
                (action.type == ActionType.OpenLong and action.stop_loss > mark_price) or \
                (action.type == ActionType.OpenShort and action.stop_loss < mark_price)
            ):
                return f"无效的止损价 (决策: {action.type.name:s}, 当前标记价格: {mark_price:.4e}, 止损价: {action.stop_loss:.4e})"

            if action.take_profit is not None and (
                action.take_profit < 0.0 or \
                (action.type == ActionType.OpenLong and action.take_profit < mark_price) or \
                (action.type == ActionType.OpenShort and action.take_profit > mark_price)
            ):
                return f"无效的止盈价 (决策: {action.type.name:s}, 当前标记价格: {mark_price:.4e}, 止盈价: {action.take_profit:.4e})"

        elif action.type in [ActionType.CloseLong, ActionType.CloseShort]:
            open_position = None
            for position in ctx.positions:
                if action.symbol == position.symbol:
                    open_position = position
                    break

            if not open_position:
                return f"{action.symbol:s}的仓位不存在"

            if (
                (action.type == ActionType.CloseLong and open_position.side == PositionSide.Short) or \
                (action.type == ActionType.CloseShort and open_position.side == PositionSide.Long)
            ):
                return f"决策 ({action.type.name:s}) 与当前{action.symbol:s}持仓冲突 ({open_position.side.name:s})"

        elif action.type == ActionType.AdjustOrder:
            open_position = None
            for position in ctx.positions:
                if action.symbol == position.symbol:
                    open_position = position
                    break

            if not open_position:
                return f"{action.symbol:s}的仓位不存在"

            mark_price = ctx.market_data[action.symbol].mark_price
            if (
                action.stop_loss < 0.0 or \
                (open_position.side == PositionSide.Long and action.stop_loss > mark_price) or \
                (open_position.side == PositionSide.Short and action.stop_loss < mark_price)
            ):
                return f"无效的止损价 (持仓方向: {open_position.side.name:s}, 当前标记价格: {mark_price:.4e}, 止损价: {action.stop_loss:.4e})"

            if action.take_profit is not None and (
                action.take_profit < 0.0 or \
                (open_position.side == PositionSide.Long and action.take_profit < mark_price) or \
                (open_position.side == PositionSide.Short and action.take_profit > mark_price)
            ):
                return f"无效的止盈价 (持仓方向: {open_position.side.name:s}, 当前标记价格: {mark_price:.4e}, 止盈价: {action.take_profit:.4e})"

        return ""
