from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading.domain.types import FLAT, LONG, SHORT
from trading.strategies.base import BaseStrategy, StrategyDecision


def _clip_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def _clip_float(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


class MeanReversionStrategy(BaseStrategy):
    strategy_id = "mean_reversion"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "bb_period": 20,
            "bb_std": 2.1,
            "z_th": 1.6,
            "rsi_low": 30.0,
            "rsi_high": 70.0,
            "stop_atr": 1.7,
            "take_atr": 2.4,
            "max_hold": 48,
        }

    @classmethod
    def sample_params(cls, rng: np.random.Generator) -> dict[str, Any]:
        return {
            "bb_period": int(rng.integers(10, 50)),
            "bb_std": float(rng.uniform(1.3, 3.2)),
            "z_th": float(rng.uniform(0.8, 3.0)),
            "rsi_low": float(rng.uniform(10.0, 40.0)),
            "rsi_high": float(rng.uniform(60.0, 90.0)),
            "stop_atr": float(rng.uniform(0.8, 3.5)),
            "take_atr": float(rng.uniform(1.2, 4.8)),
            "max_hold": int(rng.integers(8, 120)),
        }

    @classmethod
    def mutate_params(cls, params: dict[str, Any], strength: float, rng: np.random.Generator) -> dict[str, Any]:
        s = max(0.01, float(strength))
        out = dict(params)
        out["bb_period"] = _clip_int(out["bb_period"] + rng.normal(0, 6 * s), 6, 70)
        out["bb_std"] = _clip_float(out["bb_std"] + rng.normal(0, 0.25 * s), 1.0, 4.0)
        out["z_th"] = _clip_float(out["z_th"] + rng.normal(0, 0.25 * s), 0.4, 4.5)
        out["rsi_low"] = _clip_float(out["rsi_low"] + rng.normal(0, 4 * s), 5.0, 45.0)
        out["rsi_high"] = _clip_float(out["rsi_high"] + rng.normal(0, 4 * s), 55.0, 95.0)
        out["stop_atr"] = _clip_float(out["stop_atr"] + rng.normal(0, 0.35 * s), 0.5, 4.5)
        out["take_atr"] = _clip_float(out["take_atr"] + rng.normal(0, 0.5 * s), 0.6, 6.0)
        out["max_hold"] = _clip_int(out["max_hold"] + rng.normal(0, 18 * s), 6, 220)
        return out

    def decide(self, row: pd.Series, position_side: int) -> StrategyDecision:
        required = ["zscore_20", "rsi", "close", "bb_mid", "regime_adx"]
        if self.invalid_row(row, required):
            return self.flat("indicator_not_ready")

        z = float(row["zscore_20"])
        rsi = float(row["rsi"])
        z_th = float(self.params.get("z_th", 1.6))
        rsi_low = float(self.params.get("rsi_low", 30.0))
        rsi_high = float(self.params.get("rsi_high", 70.0))

        # In strong trend regime, keep mean-reversion conservative.
        strong_regime_trend = float(row["regime_adx"]) >= 30.0

        long_signal = z <= -z_th and rsi <= rsi_low and not strong_regime_trend
        short_signal = z >= z_th and rsi >= rsi_high and not strong_regime_trend
        strength = min(2.0, abs(z) / max(z_th, 1e-9))

        if position_side == FLAT:
            if long_signal:
                return StrategyDecision(LONG, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.62, "meanrev_long_entry")
            if short_signal:
                return StrategyDecision(SHORT, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.62, "meanrev_short_entry")
            return self.flat("meanrev_no_entry")

        mid = float(row["bb_mid"])
        close = float(row["close"])

        if position_side == LONG:
            if close >= mid or z >= -0.2:
                return StrategyDecision(FLAT, 0.0, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.55, "meanrev_long_exit")
            return StrategyDecision(LONG, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.35, "meanrev_hold_long")

        if close <= mid or z <= 0.2:
            return StrategyDecision(FLAT, 0.0, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.55, "meanrev_short_exit")
        return StrategyDecision(SHORT, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.35, "meanrev_hold_short")
