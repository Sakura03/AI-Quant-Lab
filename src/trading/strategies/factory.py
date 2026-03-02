from __future__ import annotations

from typing import Type

import numpy as np

from trading.strategies.base import BaseStrategy
from trading.strategies.dip_buy import DipBuyingStrategy
from trading.strategies.hybrid import HybridRegimeSwitchStrategy
from trading.strategies.mean_reversion import MeanReversionStrategy
from trading.strategies.trend import TrendFollowingStrategy


STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {
    TrendFollowingStrategy.strategy_id: TrendFollowingStrategy,
    DipBuyingStrategy.strategy_id: DipBuyingStrategy,
    MeanReversionStrategy.strategy_id: MeanReversionStrategy,
    HybridRegimeSwitchStrategy.strategy_id: HybridRegimeSwitchStrategy,
}


def build_strategy(strategy_id: str, params: dict) -> BaseStrategy:
    """Instantiate one strategy with defaults overlaid by provided params."""
    cls = STRATEGY_REGISTRY.get(strategy_id)
    if cls is None:
        raise ValueError(f"Unknown strategy_id: {strategy_id}")
    merged = cls.default_params()
    merged.update(params or {})
    return cls(merged)


def sample_strategy_params(strategy_id: str, rng: np.random.Generator) -> dict:
    """Sample one random parameter set for the selected strategy."""
    cls = STRATEGY_REGISTRY.get(strategy_id)
    if cls is None:
        raise ValueError(f"Unknown strategy_id: {strategy_id}")
    return cls.sample_params(rng)


def mutate_strategy_params(strategy_id: str, params: dict, strength: float, rng: np.random.Generator) -> dict:
    """Mutate strategy parameters while enforcing each strategy's bounds."""
    cls = STRATEGY_REGISTRY.get(strategy_id)
    if cls is None:
        raise ValueError(f"Unknown strategy_id: {strategy_id}")
    base = cls.default_params()
    base.update(params or {})
    return cls.mutate_params(base, strength, rng)
