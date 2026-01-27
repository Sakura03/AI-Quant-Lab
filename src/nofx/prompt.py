from typing import Tuple, Optional

import os.path as osp
import logging

from .logger import BaseClassWithLogger
from .enums import ActionType, PositionSide
from .structs import Balance, Context
from .utils import format_timeframe, format_time_interval


class PromptManager(BaseClassWithLogger):
    template_folder = "prompt_template"

    def __init__(self, template: str, max_positions: int, altcoin_leverage: int, BTC_ETH_leverage: int, history_span: int, run_timeframe: str, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)

        self.template = template
        self.max_positions = max_positions
        self.altcoin_leverage = altcoin_leverage
        self.BTC_ETH_leverage = BTC_ETH_leverage
        self.history_span = history_span
        self.run_timeframe = run_timeframe

        system_prompt_file = osp.join(self.template_folder, self.template + ".txt")
        assert osp.isfile(system_prompt_file)

        with open(system_prompt_file, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def generate_system_prompt(self) -> str:
        text = format_timeframe(self.run_timeframe)
        system_prompt = self.system_prompt.replace(r"{时间周期}", text)

        items = [
            f"1. 最多持仓: {self.max_positions:d}个币种 (质量>数量)",
            f"2. 杠杆限制: **山寨币最大{self.altcoin_leverage:d}x杠杆** | **BTC/ETH最大{self.BTC_ETH_leverage:d}x杠杆** (⚠️ 严格执行, 不可超过)",
            "3. 保证金: 总使用率 ≤ 95%",
            "4. 开仓金额: **≥10 USDT**",
        ]
        text = "\n".join(items)
        return system_prompt.replace(r"{添加硬约束}", text)

    def generate_user_prompt(self, ctx: Context) -> str:
        balance = ctx.balance
        positions = ctx.positions
        market_data = ctx.market_data
        strategy_data = ctx.strategy_data

        parts = []

        current_time_str = ctx.current_time.strftime("%Y-%m-%d %H:%M:%S")
        runing_time_str = format_time_interval(int(ctx.running_time.total_seconds()))
        parts.append(f"时间: {current_time_str:s} (UTC) | 周期: #{ctx.num_cycle:d} | 运行: {runing_time_str:s}")

        balance_str = balance.format()
        parts.append(f"账户信息: {balance_str:s} | 持仓数: {len(positions):d}")

        if "BTC/USDT" in market_data:
            BTC_data = market_data["BTC/USDT"]
            BTC_str = BTC_data.pct_change_str()
            parts.append(f"BTC: {BTC_data.mark_price:.4e} {BTC_str:s}")

        symbols_used = []
        if len(positions) > 0:
            parts.append("## 当前持仓")
            parts.append("需判断这些持仓是否需要提前平仓或调整止盈止损价格")
            for i, position in enumerate(positions):
                position_str = position.format(ctx.current_time)
                market_data_str = market_data[position.symbol].format(span=self.history_span)
                parts.append(f"{i+1:d}. {position_str:s}\n市场信息:\n{market_data_str:s}")
                symbols_used.append(position.symbol)

        candidates_to_open, candidates_to_reentry = [], []
        for symbol in market_data.keys():
            if symbol not in symbols_used:
                if strategy_data[symbol].action == ActionType.OpenLong:
                    candidates_to_open.append(symbol)
                elif strategy_data[symbol].state == PositionSide.Long:
                    candidates_to_reentry.append(symbol)

        if len(candidates_to_open) > 0:
            parts.append(f"## 满足开仓条件的币种 ({len(candidates_to_open):d}个)")
            parts.append("需判断这些币种是否确认开仓")
            for i, symbol in enumerate(candidates_to_open):
                market_data_str = market_data[symbol].format(span=self.history_span)
                parts.append(f"{i+1:d}. {symbol:s} 市场信息:\n{market_data_str:s}")

        if len(candidates_to_reentry) > 0:
            parts.append(f"## 满足回补条件的币种 ({len(candidates_to_reentry):d}个)")
            parts.append("需判断这些币种是否回补仓位")
            for i, symbol in enumerate(candidates_to_reentry):
                market_data_str = market_data[symbol].format(span=self.history_span)
                parts.append(f"{i+1:d}. {symbol:s} 市场信息:\n{market_data_str:s}")

        parts.extend([f"## 📊 当前夏普比率: {ctx.performance.sharpe_ratio:.2f}", "现在请分析并输出决策 (思维链 + JSON)"])

        return "\n\n".join(parts)

    def __call__(self, ctx: Context) -> Tuple[str, str]:
        system_prompt = self.generate_system_prompt()
        user_prompt = self.generate_user_prompt(ctx)
        return system_prompt, user_prompt
