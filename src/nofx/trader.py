from dataclasses import asdict
from typing import List

import os
import os.path as osp
import time
import json
import pandas as pd

from .logger import setup_logger, BaseClassWithLogger
from .config import Config
from .structs import Balance, Position, MarketData, Context, Action
from .indicator import add_indicators
from .prompt import PromptManager
from .llm_interface import LLMInterface
from .action_filter import ActionFilter
from .performance import PerformanceAnalyzer
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

        self.save_folder = osp.normpath(self.config.trader.save_folder) + "_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_folder, exist_ok=True)
        self.config.save(osp.join(self.save_folder, "config.yml"))

        # init performance analyzer
        self.performance_analyzer = PerformanceAnalyzer(self.run_timeframe)

        if self.mode == "live":
            self.start_time = pd.Timestamp.utcnow().tz_localize(None)
            # init exchange
            self.exchange = Exchange(**asdict(self.config.exchange), logger=self.logger)
            self.initial_balance = self.exchange.get_balance().total_wallet_balance
            order_restricts = {symbol: self.exchange.fetch_order_restricts(symbol) for symbol in self.symbols}
        elif self.mode == "backtest":
            self.start_time = self.config.backtest.start_time
            self.end_time = self.config.backtest.end_time
            # init backtest manager
            self.backtest_manager = BacktestManger(
                    **asdict(self.config.backtest),
                    symbols=self.symbols,
                    indicators=self.indicators,
                    analyzer=self.performance_analyzer,
                    logger=self.logger
            )
            self.initial_balance = self.config.backtest.initial_balance
            order_restricts = {symbol: (0.001, 10.0) for symbol in self.symbols}
        else:
            raise ValueError(f"不支持的模式: '{self.mode:s}'")

        # init prompt manager
        self.prompt_manager = PromptManager(**asdict(self.config.prompt), run_timeframe=self.run_timeframe, logger=self.logger)

        # init LLM interface
        self.llm_interface = LLMInterface(**asdict(self.config.llm), logger=self.logger)

        # init action filter
        self.action_filter = ActionFilter(**asdict(self.config.prompt), restricts=order_restricts, logger=self.logger)

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
    def execute_actions_backtest(self, actions: List[Action]):
        current_time = self.get_current_time()
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
            current_time = self.get_current_time()
            if current_time > self.end_time:
                break

            self.backtest_manager.tick(current_time)

            try:
                self.run_cycle()
            except Exception as e:
                self.exception(f"❌ run_cycle 异常: {e}")

        self.backtest_manager.finish()
        self.backtest_manager.analyze()
        self.backtest_manager.visualize(osp.join(self.save_folder, "viz"))

    """ Unified Interface """
    def get_current_time(self) -> pd.Timestamp:
        if self.mode == "live":
            return pd.Timestamp.utcnow().tz_localize(None)

        elif self.mode == "backtest":
            return self.start_time + pd.Timedelta(seconds=self.num_cycle * self.run_interval)

        else:
            raise ValueError(f"不支持的模式: '{self.mode:s}'")

    def get_balance(self) -> Balance:
        if self.mode == "live":
            return self.exchange.get_balance()

        elif self.mode == "backtest":
            return self.backtest_manager.get_balance()

        else:
            raise ValueError(f"不支持的模式: '{self.mode:s}'")

    def get_positions(self) -> List[Position]:
        if self.mode == "live":
            return self.exchange.get_positions()

        elif self.mode == "backtest":
            return self.backtest_manager.get_positions()

        else:
            raise ValueError(f"不支持的模式: '{self.mode:s}'")

    def get_market_data(self) -> MarketData:
        if self.mode == "live":
            market_data = self.exchange.get_market_data(self.symbols, self.timeframes)
            add_indicators(market_data, self.indicators)
            return market_data

        elif self.mode == "backtest":
            current_time = self.get_current_time()
            return self.backtest_manager.get_market_data(current_time)

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

    def dump_snapshot(self, actions: List[Action]):
        current_time = self.get_current_time()
        last_time = current_time - pd.Timedelta(seconds=self.run_interval)
        current_time_str = current_time.strftime("%Y%m%d_%H%M%S")
        to_dict = lambda x: x.to_dict()
        snapshot = {
            "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "balance": self.get_balance().to_dict(),
            "positions": list(map(to_dict, self.get_positions())),
            "actions": list(map(to_dict, actions)),
            "performance": self.performance_analyzer.get_metrics().to_dict(),
        }
        if self.mode == "backtest":
            snapshot["closed_positions"] = list(map(to_dict, self.backtest_manager.get_closed_positions(last_time, current_time)))

        os.makedirs(osp.join(self.save_folder, "snapshots"), exist_ok=True)
        with open(osp.join(self.save_folder, "snapshots", f"cycle_{self.num_cycle:05d}_{current_time_str:s}.json"), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    """ Core function """
    def run_cycle(self):
        current_time = self.get_current_time()
        ctx = Context(
            current_time=current_time,
            running_time=current_time - self.start_time,
            num_cycle=self.num_cycle,
            balance=self.get_balance(),
            positions=self.get_positions(),
            market_data=self.get_market_data(),
            performance=self.performance_analyzer.get_metrics(),
        )

        self.info("=" * 70)
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        self.info(f"*** 周期 #{self.num_cycle:d} | 当前时间: {current_time_str:s} ***")
        self.info("账户信息:")
        self.info("\t" + ctx.balance.format(self.initial_balance))
        if len(ctx.positions) > 0:
            self.info("当前持仓:")
            for i, position in enumerate(ctx.positions):
                self.info(f"\t{i+1:d}. " + position.format(current_time))

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

        # Analyze performance
        balance = self.get_balance()
        self.performance_analyzer.add_balance(current_time, balance)
        metrics = self.performance_analyzer.get_metrics()
        self.info("评测指标: " + metrics.format())
        self.info("=" * 70)

        self.dump_snapshot(actions)
