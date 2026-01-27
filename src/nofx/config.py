from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict
from typing import List, Dict, Optional, Any

import os.path as osp
import yaml
import pandas as pd

from .utils import format_symbol


class BaseConfig:
    REQUIRED_FIELDS = []
    """基础配置类，提供 from_dict / to_dict / save 等通用方法"""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if not isinstance(data, dict):
            raise TypeError(f"用于构建{cls.__name__:s}的数据类型为{type(data).__name__:s}, 应为dict")

        for key in cls.REQUIRED_FIELDS:
            if key not in data:
                raise ValueError(f"用于构建{cls.__name__:s}的dict缺失关键字: '{key:s}'")

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, BaseConfig):
                data[f.name] = value.to_dict()

        return data

    def save(self, path: str):
        """将配置保存为 YAML"""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


@dataclass
class LoggerConfig(BaseConfig):
    REQUIRED_FIELDS = ["name"]

    name: str
    log_file: Optional[str] = None
    level_name: str = "INFO"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExchangeConfig:
        config = super().from_dict(data)

        config.level_name = config.level_name.upper()

        return config


@dataclass
class ExchangeConfig(BaseConfig):
    REQUIRED_FIELDS = ["name", "api_key", "secret"]

    name: str
    api_key: str
    secret: str
    sandbox: bool = False
    enable_rate_limit: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExchangeConfig:
        config = super().from_dict(data)

        config.name = config.name.lower()

        return config


@dataclass
class PromptConfig(BaseConfig):
    REQUIRED_FIELDS = ["template", "max_positions", "altcoin_leverage", "BTC_ETH_leverage", "history_span"]

    template: str
    max_positions: int
    altcoin_leverage: int
    BTC_ETH_leverage: int
    history_span: int


@dataclass
class LLMConfig(BaseConfig):
    REQUIRED_FIELDS = ["model", "api_key"]

    model: str
    api_key: str
    temperature: float = 1.0


@dataclass
class BacktestConfig(BaseConfig):
    REQUIRED_FIELDS = ["start_time", "end_time", "data_folder"]

    start_time: pd.Timestamp
    end_time: pd.Timestamp
    data_folder: str
    initial_balance: float = 10000.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BacktestConfig:
        config = super().from_dict(data)

        config.start_time = pd.to_datetime(config.start_time, format="%Y%m%d-%H%M%S")
        config.end_time = pd.to_datetime(config.end_time, format="%Y%m%d-%H%M%S")

        return config

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()

        data["start_time"] = self.start_time.strftime("%Y%m%d-%H%M%S")
        data["end_time"] = self.end_time.strftime("%Y%m%d-%H%M%S")

        return data

@dataclass
class StrategyConfig(BaseConfig):
    REQUIRED_FIELDS = ["name", "params"]

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TraderConfig(BaseConfig):
    REQUIRED_FIELDS = ["mode", "timeframe", "save_folder", "symbols"]

    mode: str
    timeframe: str
    save_folder: str
    symbols: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TraderConfig:
        config = super().from_dict(data)

        config.mode = config.mode.lower()
        config.symbols = [format_symbol(s) for s in config.symbols]

        return config


@dataclass
class Config(BaseConfig):
    REQUIRED_FIELDS = ["logger", "prompt", "llm", "trader", "indicators"]

    logger: LoggerConfig
    prompt: PromptConfig
    llm: LLMConfig

    exchange: Optional[ExchangeConfig]
    backtest: Optional[BacktestConfig]

    strategy: StrategyConfig
    trader: TraderConfig

    indicators: Dict[str, Any] = field(default_factory=dict)

    # ---- 工厂方法 ----
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        if not isinstance(data, dict):
            raise ValueError(f"用于构建Config的数据类型为{type(data).__name__:s}, 应为dict")

        for key in cls.REQUIRED_FIELDS:
            if key not in data:
                raise ValueError(f"用于构建Config的dict缺失关键字: '{key:s}'")

        trader = TraderConfig.from_dict(data["trader"])
        if "BTC/USDT" not in trader.symbols:
            trader.symbols.append("BTC/USDT")

        if trader.mode not in ["live", "backtest"]:
            raise ValueError(f"无效的交易模式: '{trader.mode:s}'")

        extra_required = []
        if trader.mode == "live":
            extra_required.append("exchange")
        elif trader.mode == "backtest":
            extra_required.append("backtest")

        for key in extra_required:
            if key not in data:
                raise ValueError(f"用于构建Config的dict缺失关键字: '{key:s}'")

        exchange = ExchangeConfig.from_dict(data["exchange"]) if "exchange" in data else None
        backtest = BacktestConfig.from_dict(data["backtest"]) if "backtest" in data else None
        strategy = StrategyConfig.from_dict(data["strategy"]) if "strategy" in data else None
        indicators = cls.resolve_indicators(data.get("indicators", {}))

        return Config(
            logger=LoggerConfig.from_dict(data["logger"]),
            prompt=PromptConfig.from_dict(data["prompt"]),
            llm=LLMConfig.from_dict(data["llm"]),
            exchange=exchange,
            backtest=backtest,
            trader=trader,
            strategy=strategy,
            indicators=indicators,
        )

    @staticmethod
    def resolve_indicators(indicators: Dict[str, Any]) -> Dict[str, Any]:
        """支持字符串引用的指标配置，例如：
        indicators:
            1m: [{name: ema, period: 10}]
            5m: "1m"
            1h: [{name: ema, period: 30}]
        """
        def resolve(key: str, visited: set[str]) -> Any:
            if key in visited:
                raise ValueError(f"不同时间周期的技术指标参数存在循环引用")
            visited.add(key)
            val = indicators.get(key)
            if isinstance(val, str):
                return resolve(val, visited)
            return val

        resolved = {}
        for k in indicators:
            params = resolve(k, set())
            for param in params:
                required = ["name", "params", "display_name"]
                for key in required:
                    if key not in param:
                        raise ValueError(f"技术指标参数缺失关键字: '{key:s}'")

                param["name"] = param["name"].upper()

            resolved[k] = params

        return resolved

    # ---- YAML 文件解析 ----
    @staticmethod
    def parse_config_file(config_path: str) -> Config:
        if not osp.exists(config_path):
            raise FileNotFoundError(f"文件不存在: {config_path:s}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"无效的YAML格式: {config_path:s}")

        config = Config.from_dict(config)
        config.validate()
        return config

    # ---- 校验逻辑 ----
    def validate(self):
        """进行配置合法性检查"""
        if not (self.exchange.api_key and self.exchange.secret and self.llm.api_key):
            raise ValueError("必须提供交易所的API key和密钥和大模型的API key")
        if self.prompt.BTC_ETH_leverage < 1 or self.prompt.altcoin_leverage < 1:
            raise ValueError(f"杠杆倍数必须为正, 当前BTC/ETH杠杆: {self.prompt.BTC_ETH_leverage:d}x, 山寨币杠杆: {self.prompt.altcoin_leverage:d}x")
        if self.prompt.max_positions < 1:
            raise ValueError(f"最大持仓数必须为正, 当前: {self.prompt.max_positions:d}")
        if len(self.trader.symbols) == 0:
            raise ValueError("交易对列表为空, 请提供至少一个交易对")
        if len(self.indicators) == 0:
            raise ValueError("技术指标为空, 请提供至少一个时间周期 (该时间周期下无技术指标也可)")
