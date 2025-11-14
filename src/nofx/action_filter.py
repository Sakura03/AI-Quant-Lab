from typing import List, Optional

import logging

from .logger import BaseClassWithLogger
from .enums import PositionSide, ActionType
from .structs import Action, Context


class ActionFilter(BaseClassWithLogger):
    priority_map = {
        ActionType.CloseLong: 0,
        ActionType.CloseShort: 0,
        ActionType.OpenLong: 1,
        ActionType.OpenShort: 1,
        ActionType.DoNothing: 2,
    }

    def __init__(self, r_ratio: float, altcoin_leverage: int, BTC_ETH_leverage: int, max_positions: int, logger: Optional[logging.Logger] = None, **kwargs):
        super().__init__(logger=logger)

        self.r_ratio = r_ratio * 0.6  # Hack here: do not restrict R-ratio too strictly
        self.altcoin_leverage = altcoin_leverage
        self.BTC_ETH_leverage = BTC_ETH_leverage
        self.max_positions = max_positions

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

        if action.type == ActionType.DoNothing:
            return ""

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

            if available_balance < action.position_size_usd / action.leverage:
                return f"保证金不足 (可用保证金: {available_balance:.2f} USDT, 仓位: {action.position_size_usd:.2f} USDT, 杠杆: {action.leverage:d}x)"

            if action.stop_loss < 0.0 or action.take_profit < 0.0:
                return f"止损价或止盈价为负 (止损价: {action.stop_loss:.4e}, 止盈价: {action.stop_loss:.4e})"

            mark_price = ctx.market_data[action.symbol].mark_price
            if (
                (action.type == ActionType.OpenLong and not action.stop_loss < mark_price < action.take_profit) or \
                (action.type == ActionType.OpenShort and not action.take_profit < mark_price < action.stop_loss)
            ):
                return f"止损价或止盈价无效 (决策: {action.type.name:s}, 当前标记价格: {mark_price:.4e}, 止损价: {action.stop_loss:.4e}, 止盈价: {action.take_profit:.4e})"

            risk_pct = abs(action.stop_loss - mark_price) / mark_price
            reward_pct = abs(mark_price - action.take_profit) / mark_price
            r_ratio = reward_pct / risk_pct
            if r_ratio < self.r_ratio:
                return f"盈亏比过低 (决策: {action.type.name:s}, 当前标记价格: {mark_price:.4e}, 止损价: {action.stop_loss:.4e}, 止盈价: {action.take_profit:.4e}, 潜在亏损: {risk_pct*100:.1f}%, 潜在盈利: {reward_pct*100:.1f}%, 盈亏比: {r_ratio:.2f})"

        if action.type in [ActionType.CloseLong, ActionType.CloseShort]:
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

        return ""
