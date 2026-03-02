from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import os.path as osp

import pandas as pd
import yaml


TIME_FMT = "%Y%m%d-%H%M%S"
VALID_STRATEGIES = {"trend_following", "dip_buying", "mean_reversion", "hybrid_regime_switch"}


def parse_timestamp(value: Any) -> pd.Timestamp:
    """Parse a flexible timestamp input into timezone-naive pandas Timestamp."""
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(None) if value.tzinfo is not None else value
    if isinstance(value, str):
        try:
            return pd.to_datetime(value, format=TIME_FMT)
        except ValueError:
            return pd.to_datetime(value)
    return pd.to_datetime(value)


@dataclass
class EngineConfig:
    """Engine-level settings such as random seed and job parallelism."""

    seed: int = 42
    n_jobs: int = 1


@dataclass
class DataConfig:
    """Data source and time-range configuration for backtest/optimize."""

    root: str = "data"
    universe: list[str] = None
    exec_tf: str = "1m"
    signal_tfs: list[str] = None
    regime_tfs: list[str] = None
    start: pd.Timestamp = pd.Timestamp("2020-01-01")
    end: pd.Timestamp = pd.Timestamp("2026-02-01")
    warmup_bars: int = 600
    timestamp_semantics: str = "close"

    def __post_init__(self):
        """Populate default universe/timeframes when config omits them."""
        if self.universe is None:
            self.universe = ["BTC/USDT", "ETH/USDT"]
        if self.signal_tfs is None:
            self.signal_tfs = ["1h"]
        if self.regime_tfs is None:
            self.regime_tfs = ["4h", "1d"]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataConfig":
        """Build DataConfig from dict and normalize start/end timestamps."""
        cfg = cls(**data)
        cfg.start = parse_timestamp(cfg.start)
        cfg.end = parse_timestamp(cfg.end)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize DataConfig to plain dict with stable timestamp format."""
        out = asdict(self)
        out["start"] = self.start.strftime(TIME_FMT)
        out["end"] = self.end.strftime(TIME_FMT)
        return out


@dataclass
class SplitConfig:
    """Walk-forward split lengths and embargo gap."""

    train_months: int = 18
    val_months: int = 3
    test_months: int = 3
    step_months: int = 3
    embargo_days: int = 3


@dataclass
class StrategyConfig:
    """Strategy universe and per-strategy position cap."""

    enabled: list[str] = None
    per_strategy_max_positions: int = 3

    def __post_init__(self):
        """Apply full default strategy set when `enabled` is not specified."""
        if self.enabled is None:
            self.enabled = [
                "trend_following",
                "dip_buying",
                "mean_reversion",
                "hybrid_regime_switch",
            ]


@dataclass
class PortfolioConfig:
    """Portfolio risk and exposure constraints."""

    initial_balance: float = 100000.0
    max_gross_leverage: float = 1.8
    max_total_positions: int = 8
    max_symbol_weight: float = 0.35
    target_vol_annual: float = 0.18
    dd_soft: float = 0.08
    dd_hard: float = 0.15
    min_risk_scale: float = 0.2
    risk_per_trade: float = 0.01


@dataclass
class CostsConfig:
    """Trading frictions and funding settings."""

    fee_rate: float = 0.0005
    slippage_bps: float = 2.0
    funding_enabled: bool = True


@dataclass
class OptimizerConfig:
    """Evolutionary optimizer hyper-parameters."""

    trials_per_window: int = 800
    population: int = 80
    elite_top_k: int = 12
    prior_adoption_prob: float = 0.7
    mutation_strength: float = 0.2
    crossover_prob: float = 0.3
    min_trades_per_window: int = 20


@dataclass
class ObjectiveConfig:
    """Objective weights and hard constraints used in optimization."""

    weights: dict[str, float] = None
    hard_limits: dict[str, float] = None

    def __post_init__(self):
        """Fill default objective weights/limits when they are omitted."""
        if self.weights is None:
            self.weights = {
                "sharpe": 1.0,
                "max_drawdown": 2.2,
                "turnover": 0.12,
                "overfit_gap": 0.2,
                "concentration": 0.5,
            }
        if self.hard_limits is None:
            self.hard_limits = {"max_drawdown": 0.15}


@dataclass
class OutputConfig:
    """Output directory configuration."""

    dir: str = "results/trading_enterprise"


@dataclass
class TradingConfig:
    """Top-level runtime configuration for the trading framework."""

    engine: EngineConfig
    data: DataConfig
    split: SplitConfig
    strategies: StrategyConfig
    portfolio: PortfolioConfig
    costs: CostsConfig
    optimizer: OptimizerConfig
    objective: ObjectiveConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TradingConfig":
        """Construct a full TradingConfig object tree from raw mapping."""
        cfg = cls(
            engine=EngineConfig(**raw.get("engine", {})),
            data=DataConfig.from_dict(raw.get("data", {})),
            split=SplitConfig(**raw.get("split", {})),
            strategies=StrategyConfig(**raw.get("strategies", {})),
            portfolio=PortfolioConfig(**raw.get("portfolio", {})),
            costs=CostsConfig(**raw.get("costs", {})),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            objective=ObjectiveConfig(**raw.get("objective", {})),
            output=OutputConfig(**raw.get("output", {})),
        )
        cfg.validate()
        return cfg

    @staticmethod
    def parse_file(path: str) -> "TradingConfig":
        """Load YAML config file and parse it into TradingConfig."""
        if not osp.isfile(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise TypeError("Config root must be a mapping")
        return TradingConfig.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """Serialize full config to dict for persistence and cloning."""
        return {
            "engine": asdict(self.engine),
            "data": self.data.to_dict(),
            "split": asdict(self.split),
            "strategies": asdict(self.strategies),
            "portfolio": asdict(self.portfolio),
            "costs": asdict(self.costs),
            "optimizer": asdict(self.optimizer),
            "objective": asdict(self.objective),
            "output": asdict(self.output),
        }

    def validate(self):
        """Run schema-level and risk-related sanity checks."""
        if self.data.timestamp_semantics != "close":
            raise ValueError("data.timestamp_semantics must be 'close'")
        if self.data.start >= self.data.end:
            raise ValueError("data.start must be earlier than data.end")
        if not self.data.universe:
            raise ValueError("data.universe cannot be empty")
        if not self.data.signal_tfs:
            raise ValueError("data.signal_tfs cannot be empty")
        if self.data.exec_tf in self.data.signal_tfs:
            pass
        if self.portfolio.initial_balance <= 0:
            raise ValueError("portfolio.initial_balance must be positive")
        if self.portfolio.max_gross_leverage <= 0:
            raise ValueError("portfolio.max_gross_leverage must be positive")
        if not (0 < self.portfolio.max_symbol_weight <= 1):
            raise ValueError("portfolio.max_symbol_weight must be in (0, 1]")
        if not (0 <= self.portfolio.dd_soft <= self.portfolio.dd_hard <= 1):
            raise ValueError("Require 0 <= dd_soft <= dd_hard <= 1")
        if not (0 < self.portfolio.min_risk_scale <= 1):
            raise ValueError("portfolio.min_risk_scale must be in (0, 1]")
        if self.optimizer.population <= 0:
            raise ValueError("optimizer.population must be positive")
        if self.optimizer.trials_per_window <= 0:
            raise ValueError("optimizer.trials_per_window must be positive")
        if self.optimizer.elite_top_k <= 0:
            raise ValueError("optimizer.elite_top_k must be positive")
        if self.optimizer.elite_top_k > self.optimizer.population:
            raise ValueError("optimizer.elite_top_k cannot exceed optimizer.population")
        if self.split.step_months <= 0:
            raise ValueError("split.step_months must be positive")
        if self.split.train_months <= 0 or self.split.val_months <= 0 or self.split.test_months <= 0:
            raise ValueError("train/val/test months must be positive")
        if self.costs.fee_rate < 0 or self.costs.slippage_bps < 0:
            raise ValueError("costs fee/slippage must be non-negative")
        invalid = [x for x in self.strategies.enabled if x not in VALID_STRATEGIES]
        if invalid:
            raise ValueError(f"Unsupported strategy IDs: {invalid}")
        hard_mdd = self.objective.hard_limits.get("max_drawdown", 0.15)
        if not (0 < hard_mdd <= 1):
            raise ValueError("objective.hard_limits.max_drawdown must be in (0, 1]")
