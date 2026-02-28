from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import copy
import os.path as osp

import pandas as pd
import yaml


TIME_FMT = "%Y%m%d-%H%M%S"


def parse_timestamp(value: Any) -> pd.Timestamp:
    """将配置中的时间值统一转成无时区的 `pd.Timestamp`。"""
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(None) if value.tzinfo is not None else value
    if isinstance(value, str):
        return pd.to_datetime(value, format=TIME_FMT)
    return pd.to_datetime(value)


@dataclass
class EngineConfig:
    """运行引擎配置。当前仅支持回测模式。"""
    mode: str = "backtest"
    seed: int = 42


@dataclass
class DataConfig:
    """数据加载相关配置。"""
    data_folder: str
    symbols: list[str]
    timeframe_signal: str = "1h"
    timeframe_execution: str = "1m"
    start_time: pd.Timestamp = pd.Timestamp("2025-01-01")
    end_time: pd.Timestamp = pd.Timestamp("2025-07-01")
    warmup_bars: int = 300

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        """从字典构造数据配置，并解析起止时间字段。"""
        config = cls(**data)
        config.start_time = parse_timestamp(config.start_time)
        config.end_time = parse_timestamp(config.end_time)
        return config

    def to_dict(self) -> Dict[str, Any]:
        """将数据配置序列化为可写入 YAML 的字典。"""
        data = asdict(self)
        data["start_time"] = self.start_time.strftime(TIME_FMT)
        data["end_time"] = self.end_time.strftime(TIME_FMT)
        return data


@dataclass
class CostsConfig:
    """交易成本配置，包括手续费、滑点和资金费开关。"""
    fee_rate: float = 0.0005
    slippage_bps: float = 2.0
    funding_rate_enabled: bool = True


@dataclass
class RiskConfig:
    """风控参数配置。"""
    initial_balance: float = 10000.0
    max_gross_leverage: float = 2.0
    risk_per_trade: float = 0.01
    max_positions: int = 3
    stop_atr_mult: float = 2.0
    take_atr_mult: float = 3.0


@dataclass
class StrategyConfig:
    """策略名称与参数配置。"""
    name: str = "hybrid_trend_meanrev"
    params: Dict[str, Any] | None = None


@dataclass
class OptimizationConfig:
    """参数优化配置。"""
    enabled: bool = True
    method: str = "walk_forward_random_search"
    trials: int = 600
    train_months: int = 6
    valid_months: int = 2
    test_months: int = 2
    drawdown_limit: float = 0.2
    objective_weights: Dict[str, float] | None = None

    def __post_init__(self):
        """在未提供权重时填充默认目标函数权重。"""
        if self.objective_weights is None:
            self.objective_weights = {
                "sharpe": 1.0,
                "max_drawdown_penalty": 2.0,
                "turnover_penalty": 0.1,
            }


@dataclass
class OutputConfig:
    """输出目录配置。"""
    save_folder: str = "results/trading"


@dataclass
class TradingConfig:
    """交易系统总配置对象。"""
    engine: EngineConfig
    data: DataConfig
    costs: CostsConfig
    risk: RiskConfig
    strategy: StrategyConfig
    optimization: OptimizationConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingConfig":
        """从原始字典构建并校验总配置。"""
        if not isinstance(data, dict):
            raise TypeError("配置必须是 dict")

        cfg = cls(
            engine=EngineConfig(**data.get("engine", {})),
            data=DataConfig.from_dict(data["data"]),
            costs=CostsConfig(**data.get("costs", {})),
            risk=RiskConfig(**data.get("risk", {})),
            strategy=StrategyConfig(**data.get("strategy", {})),
            optimization=OptimizationConfig(**data.get("optimization", {})),
            output=OutputConfig(**data.get("output", {})),
        )
        cfg.validate()
        return cfg

    @staticmethod
    def parse_config_file(path: str) -> "TradingConfig":
        """从 YAML 文件读取配置并解析成 `TradingConfig`。"""
        if not osp.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TradingConfig.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """将总配置转换为可序列化字典。"""
        return {
            "engine": asdict(self.engine),
            "data": self.data.to_dict(),
            "costs": asdict(self.costs),
            "risk": asdict(self.risk),
            "strategy": asdict(self.strategy),
            "optimization": asdict(self.optimization),
            "output": asdict(self.output),
        }

    def validate(self):
        """执行关键字段校验，防止运行期出现低级配置错误。"""
        if self.engine.mode != "backtest":
            raise ValueError("当前 trading 模块仅支持 backtest 模式")
        if self.data.start_time >= self.data.end_time:
            raise ValueError("data.start_time 必须早于 data.end_time")
        if len(self.data.symbols) == 0:
            raise ValueError("data.symbols 不能为空")
        if self.risk.initial_balance <= 0:
            raise ValueError("risk.initial_balance 必须大于 0")
        if self.risk.max_positions <= 0:
            raise ValueError("risk.max_positions 必须大于 0")
        if self.risk.max_gross_leverage <= 0:
            raise ValueError("risk.max_gross_leverage 必须大于 0")
        if self.risk.risk_per_trade <= 0:
            raise ValueError("risk.risk_per_trade 必须大于 0")
        if self.costs.fee_rate < 0 or self.costs.slippage_bps < 0:
            raise ValueError("fee_rate/slippage_bps 不能小于 0")
        if self.strategy.name != "hybrid_trend_meanrev":
            raise ValueError("当前仅支持 strategy.name=hybrid_trend_meanrev")


def clone_config_with_period(
    config: TradingConfig,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    strategy_params: Optional[Dict[str, Any]] = None,
) -> TradingConfig:
    """基于现有配置创建一个新的时间窗口配置，可选覆盖策略参数。"""
    new_cfg = TradingConfig.from_dict(copy.deepcopy(config.to_dict()))
    new_cfg.data.start_time = parse_timestamp(start_time)
    new_cfg.data.end_time = parse_timestamp(end_time)
    if strategy_params is not None:
        new_cfg.strategy.params = dict(strategy_params)
    return new_cfg
