from dataclasses import asdict
from typing import List

import time
import pandas as pd

from .logger import setup_logger, BaseClassWithLogger
from .config import Config
from .structs import Context, Action
from .indicator import add_indicators
from .prompt import PromptManager
from .llm_interface import LLMInterface
from .action_filter import ActionFilter
from .exchange import Exchange
from .backtest import BacktestManger
from .utils import timeframe_to_seconds, format_time_interval


class AutoTrader(BaseClassWithLogger):
    def __init__(self, config_path: str):
        # parse config
        self.config = Config.parse_config_file(config_path)

        # setup logger
        logger = setup_logger(**asdict(self.config.logger))
        super().__init__(logger=logger)

        self.symbols = self.config.trader.symbols
        self.indicators = self.config.indicators
        self.timeframes = list(self.indicators.keys())

        self.mode = self.config.trader.mode
        self.run_timeframe = self.config.trader.timeframe
        self.run_interval = timeframe_to_seconds(self.run_timeframe)
        self.num_cycle = 0

        # init prompt manager
        self.prompt_manager = PromptManager(**asdict(self.config.prompt), run_timeframe=self.run_timeframe, logger=self.logger)

        # init LLM interface
        self.llm_interface = LLMInterface(**asdict(self.config.llm), logger=self.logger)

        # init action filter
        self.action_filter = ActionFilter(**asdict(self.config.prompt), logger=self.logger)

        if self.mode == "live":
            self.start_time = pd.Timestamp.utcnow().tz_localize(None)
            # init exchange
            self.exchange = Exchange(**asdict(self.config.exchange), logger=self.logger)
            self.initial_balance = self.exchange.get_balance().total_wallet_balance
        elif self.mode == "backtest":
            self.start_time = self.config.backtest.start_time
            self.end_time = self.config.backtest.end_time
            # init backtest manager
            self.backtest_manager = BacktestManger(**asdict(self.config.backtest), symbols=self.symbols, indicators=self.indicators, logger=self.logger)
            self.initial_balance = self.config.backtest.initial_balance
        else:
            raise ValueError(f"不支持的模式: '{self.mode:s}'")

    """ Live Trader """
    def next_run_time(self) -> pd.Timestamp:
        """
            计算下一个整周期时间
            例如: timeframe = 1h, 则返回下一个整点 (00:00, 01:00, ...)
        """
        now = pd.Timestamp.utcnow().tz_localize(None)
        seconds = self.run_interval
        next_run = (now.floor(f"{seconds}s") + pd.Timedelta(seconds=seconds))
        return next_run

    def build_context_live(self) -> Context:
        # Balance and positions
        balance = self.exchange.get_balance()
        positions = self.exchange.get_positions()

        # Market data
        market_data = self.exchange.get_market_data(self.symbols, self.timeframes)
        add_indicators(market_data, self.indicators)

        # Create context
        current_time = pd.Timestamp.utcnow().tz_localize(None)
        return Context(
            current_time=current_time,
            running_time=current_time - self.start_time,
            num_cycle=self.num_cycle,
            balance=balance,
            positions=positions,
            market_data=market_data,
        )

    def execute_actions_live(self, actions: List[Action]):
        self.exchange.execute_actions(actions)

    def run_live(self):
        """
            无限循环运行 run_cycle()，自动对齐到 timeframe 的边界
        """
        self.info(f"🕒 交易算法以时间周期{self.run_timeframe:s}运行, 账户初始金额: {self.initial_balance:.2f} USDT")

        while True:
            self.num_cycle += 1
            next_run = self.next_run_time()
            sleep_seconds = (next_run - pd.Timestamp.utcnow().tz_localize(None)).total_seconds()
            if sleep_seconds > 0:
                sleep_time_str = format_time_interval(int(sleep_seconds), return_second=True)
                next_run_str = next_run.strftime("%Y-%m-%d %H:%M:%S")
                self.info(f"⏳ 等待{sleep_time_str:s}, 直到下个周期: {next_run_str:s}")
                time.sleep(sleep_seconds)

            try:
                self.run_cycle()
            except Exception as e:
                self.exception(f"❌ run_cycle 异常: {e}")
                time.sleep(5)  # 短暂冷却防止异常死循环

    """ Backtester """
    def build_context_backtest(self) -> Context:
        current_time = self.start_time + pd.Timedelta(seconds=self.num_cycle * self.run_interval)
        self.backtest_manager.tick(current_time)

        # Balance and positions
        balance = self.backtest_manager.get_balance()
        positions = self.backtest_manager.get_positions()

        # Market data
        market_data = self.backtest_manager.get_market_data(current_time)
        add_indicators(market_data, self.indicators)

        # Create context
        return Context(
            current_time=current_time,
            running_time=current_time - self.start_time,
            num_cycle=self.num_cycle,
            balance=balance,
            positions=positions,
            market_data=market_data,
        )

    def execute_actions_backtest(self, actions: List[Action]):
        current_time = self.start_time + pd.Timedelta(seconds=self.num_cycle * self.run_interval)
        self.backtest_manager.execute_actions(actions, current_time)

    def run_backtest(self):
        """
            回测模式
        """
        start_time_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = self.end_time.strftime("%Y-%m-%d %H:%M:%S")
        self.info(f"回测时间段: {start_time_str:s}-{end_time_str:s}, 时间周期: {self.run_timeframe:s}, 账户初始金额: {self.initial_balance:.2f} USDT")

        while True:
            self.num_cycle += 1
            current_time = self.start_time + pd.Timedelta(seconds=self.num_cycle * self.run_interval)
            if current_time > self.end_time:
                break

            try:
                self.run_cycle()
            except Exception as e:
                self.exception(f"❌ run_cycle 异常: {e}")

        self.backtest_manager.finish()
        self.backtest_manager.analyze()

    """ Unified Interface """
    def build_context(self) -> Context:
        if self.mode == "live":
            return self.build_context_live()

        elif self.mode == "backtest":
            return self.build_context_backtest()

        else:
            raise ValueError(f"不支持的模式: '{self.mode:s}'")

    def execute_actions(self, actions: List[Action]):
        if self.mode == "live":
            self.execute_actions_live(actions)

        elif self.mode == "backtest":
            self.execute_actions_backtest(actions)

        else:
            raise ValueError(f"不支持的模式: '{self.mode:s}'")

    def run(self):
        if self.mode == "live":
            self.run_live()

        elif self.mode == "backtest":
            self.run_backtest()

        else:
            raise ValueError(f"不支持的模式: '{self.mode:s}'")

    """ Core function """
    def run_cycle(self):
        ctx = self.build_context()

        self.info("=" * 70)
        current_time_str = ctx.current_time.strftime("%Y-%m-%d %H:%M:%S")
        self.info(f"*** 周期 #{self.num_cycle:d} | 当前时间: {current_time_str:s} ***")
        self.info("账户信息:")
        self.info("\t" + ctx.balance.format(self.initial_balance))
        if len(ctx.positions) > 0:
            self.info("当前持仓:")
            for i, position in enumerate(ctx.positions):
                self.info(f"\t{i+1:d}. " + position.format())

        # Generate system and user prompt
        system_prompt, user_prompt = self.prompt_manager(ctx)
        self.debug("系统提示词:\n" + system_prompt)
        self.debug("用户提示词:\n" + user_prompt)

        # Call LLM and parse results
        reasoning, actions = self.llm_interface(system_prompt, user_prompt)
        self.info("思维链 (CoT):\n" + reasoning)

        # Filter actions
        actions = self.action_filter(actions, ctx)
        if len(actions) > 0:
            self.info("AI决策:")
            for i, action in enumerate(actions):
                self.info(f"\t{i+1:d}. " + action.format())

        # Execute actions
        self.execute_actions(actions)
        self.info("=" * 70)
