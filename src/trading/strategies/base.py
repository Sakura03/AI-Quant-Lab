from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading.domain.types import FLAT


@dataclass
class StrategyDecision:
    side: int
    strength: float
    stop_atr: float
    take_atr: float
    confidence: float
    reason: str


class BaseStrategy:
    strategy_id: str = "base"

    def __init__(self, params: dict[str, Any]):
        self.params = params

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def sample_params(cls, rng: np.random.Generator) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def mutate_params(cls, params: dict[str, Any], strength: float, rng: np.random.Generator) -> dict[str, Any]:
        raise NotImplementedError

    def decide(self, row: pd.Series, position_side: int) -> StrategyDecision:
        raise NotImplementedError

    @staticmethod
    def invalid_row(row: pd.Series, required: list[str]) -> bool:
        return any(pd.isna(row.get(c)) for c in required)

    @staticmethod
    def flat(reason: str = "no_signal") -> StrategyDecision:
        return StrategyDecision(side=FLAT, strength=0.0, stop_atr=2.0, take_atr=3.0, confidence=0.0, reason=reason)

    @staticmethod
    def signed_strength(strength: float, side: int) -> float:
        return abs(strength) * (1 if side > 0 else -1 if side < 0 else 0)
