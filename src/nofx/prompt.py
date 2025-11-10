from typing import Tuple, Optional

import os.path as osp
import logging

from .logger import BaseClassWithLogger
from .structs import Balance, Context
from .utils import format_timeframe, format_time_interval


class PromptManager(BaseClassWithLogger):
    template_folder = "prompt_template"

    def __init__(self, template: str, r_ratio: float, max_positions: int, altcoin_leverage: int, BTC_ETH_leverage: int, history_span: int, run_timeframe: str, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)

        self.template = template
        self.r_ratio = r_ratio
        self.max_positions = max_positions
        self.altcoin_leverage = altcoin_leverage
        self.BTC_ETH_leverage = BTC_ETH_leverage
        self.history_span = history_span
        self.run_timeframe = run_timeframe

        system_prompt_file = osp.join(self.template_folder, self.template + ".txt")
        if not osp.isfile(system_prompt_file):
            self.template = "default"
            system_prompt_file = osp.join(self.template_folder, "default.txt")
            self.warning("模板不存在, 使用default模板")
        assert osp.isfile(system_prompt_file)

        with open(system_prompt_file, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def generate_system_prompt(self, balance: Balance) -> str:
        text = format_timeframe(self.run_timeframe)
        system_prompt = self.system_prompt.replace(r"{时间周期}", text)

        BTC_ETH_position_size = balance.total_wallet_balance * self.BTC_ETH_leverage / self.max_positions
        altcoin_position_size = balance.total_wallet_balance * self.altcoin_leverage / self.max_positions
        items = [
            f"1. 盈亏比: 必须 ≥ {self.r_ratio:.2f} (冒1%风险，赚{self.r_ratio:.2f}%+收益)",
            f"2. 最多持仓: {self.max_positions:d}个币种 (质量>数量)",
            f"3. 单币仓位: BTC/ETH {BTC_ETH_position_size*0.8:.2f}-{BTC_ETH_position_size:.2f} USDT | 山寨币{altcoin_position_size*0.8:.2f}-{altcoin_position_size:.2f} USDT",
            f"4. 杠杆限制: **山寨币最大{self.altcoin_leverage:d}x杠杆** | **BTC/ETH最大{self.BTC_ETH_leverage:d}x杠杆** (⚠️ 严格执行，不可超过)",
            "5. 保证金: 总使用率 ≤ 95%",
            "6. 开仓金额: **≥10 USDT**",
        ]
        text = "\n".join(items)
        return system_prompt.replace(r"{添加硬约束}", text)

    def generate_user_prompt(self, ctx: Context) -> str:
        balance = ctx.balance
        positions = ctx.positions
        market_data = ctx.market_data

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
            for i, position in enumerate(positions):
                position_str = position.format()
                market_data_str = market_data[position.symbol].format(span=self.history_span)
                parts.append(f"{i+1:d}. {position_str:s}\n市场信息:\n{market_data_str:s}")
                symbols_used.append(position.symbol)

        candidates = []
        for symbol in market_data.keys():
            if symbol not in symbols_used:
                candidates.append(symbol)

        if len(candidates) > 0:
            parts.append(f"## 其它候选币种 ({len(candidates):d}个)")
            for i, symbol in enumerate(candidates):
                market_data_str = market_data[symbol].format(span=self.history_span)
                parts.append(f"{i+1:d}. {symbol:s} 市场信息:\n{market_data_str:s}")

        #TODO(Xinyu): add Sharpe ratio analysis
        # parts.extend([f"## 📊 当前夏普比率: %.2f", "现在请分析并输出决策 (思维链 + JSON)"])
        parts.append("现在请分析并输出决策 (思维链 + JSON)")

        return "\n\n".join(parts)

    def __call__(self, ctx: Context) -> Tuple[str, str]:
        system_prompt = self.generate_system_prompt(ctx.balance)
        user_prompt = self.generate_user_prompt(ctx)
        return system_prompt, user_prompt
