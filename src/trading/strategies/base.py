from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading.domain.types import FLAT


@dataclass
class StrategyDecision:
    """Standardized strategy output consumed by the execution engine."""

    side: int
    strength: float
    stop_atr: float
    take_atr: float
    confidence: float
    reason: str


class BaseStrategy:
    """Abstract base class for all strategy implementations."""

    strategy_id: str = "base"

    def __init__(self, params: dict[str, Any]):
        """Store validated strategy parameters."""
        self.params = params

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Return default parameter dictionary."""
        raise NotImplementedError

    @classmethod
    def sample_params(cls, rng: np.random.Generator) -> dict[str, Any]:
        """Sample one random parameter set for optimization."""
        raise NotImplementedError

    @classmethod
    def mutate_params(cls, params: dict[str, Any], strength: float, rng: np.random.Generator) -> dict[str, Any]:
        """Mutate one parameter set with a tunable mutation strength."""
        raise NotImplementedError

    def decide(self, row: pd.Series, position_side: int) -> StrategyDecision:
        """Generate strategy action from one feature row and current side."""
        raise NotImplementedError

    @staticmethod
    def invalid_row(row: pd.Series, required: list[str]) -> bool:
        """Check whether any required feature is missing."""
        return any(pd.isna(row.get(c)) for c in required)

    @staticmethod
    def flat(reason: str = "no_signal") -> StrategyDecision:
        """Return a no-position decision."""
        return StrategyDecision(side=FLAT, strength=0.0, stop_atr=2.0, take_atr=3.0, confidence=0.0, reason=reason)

    @staticmethod
    def signed_strength(strength: float, side: int) -> float:
        """Convert unsigned signal strength into signed value by side."""
        return abs(strength) * (1 if side > 0 else -1 if side < 0 else 0)
