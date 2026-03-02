from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading.domain.types import FLAT, LONG, SHORT
from trading.strategies.base import BaseStrategy, StrategyDecision


def _clip_int(v: float, lo: int, hi: int) -> int:
    """Clamp and round to bounded integer."""
    return int(max(lo, min(hi, round(v))))


def _clip_float(v: float, lo: float, hi: float) -> float:
    """Clamp to bounded float."""
    return float(max(lo, min(hi, v)))


class DipBuyingStrategy(BaseStrategy):
    """Dip/spike reversal strategy with regime-side safety checks."""

    strategy_id = "dip_buying"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Return default hyper-parameters."""
        return {
            "dip_pct_th": 0.035,
            "spike_pct_th": 0.035,
            "rsi_low": 28.0,
            "rsi_high": 72.0,
            "rebound_exit": 0.02,
            "stop_atr": 1.8,
            "take_atr": 2.8,
            "max_hold": 36,
        }

    @classmethod
    def sample_params(cls, rng: np.random.Generator) -> dict[str, Any]:
        """Sample one random parameter set for optimizer initialization."""
        return {
            "dip_pct_th": float(rng.uniform(0.01, 0.08)),
            "spike_pct_th": float(rng.uniform(0.01, 0.08)),
            "rsi_low": float(rng.uniform(10.0, 35.0)),
            "rsi_high": float(rng.uniform(65.0, 90.0)),
            "rebound_exit": float(rng.uniform(0.005, 0.04)),
            "stop_atr": float(rng.uniform(1.0, 3.5)),
            "take_atr": float(rng.uniform(1.2, 4.0)),
            "max_hold": int(rng.integers(6, 80)),
        }

    @classmethod
    def mutate_params(cls, params: dict[str, Any], strength: float, rng: np.random.Generator) -> dict[str, Any]:
        """Mutate parameters with bounded gaussian noise."""
        s = max(0.01, float(strength))
        out = dict(params)
        out["dip_pct_th"] = _clip_float(out["dip_pct_th"] + rng.normal(0, 0.01 * s), 0.005, 0.15)
        out["spike_pct_th"] = _clip_float(out["spike_pct_th"] + rng.normal(0, 0.01 * s), 0.005, 0.15)
        out["rsi_low"] = _clip_float(out["rsi_low"] + rng.normal(0, 4 * s), 5.0, 45.0)
        out["rsi_high"] = _clip_float(out["rsi_high"] + rng.normal(0, 4 * s), 55.0, 95.0)
        out["rebound_exit"] = _clip_float(out["rebound_exit"] + rng.normal(0, 0.006 * s), 0.002, 0.08)
        out["stop_atr"] = _clip_float(out["stop_atr"] + rng.normal(0, 0.4 * s), 0.6, 4.5)
        out["take_atr"] = _clip_float(out["take_atr"] + rng.normal(0, 0.5 * s), 0.8, 6.0)
        out["max_hold"] = _clip_int(out["max_hold"] + rng.normal(0, 12 * s), 4, 160)
        return out

    def decide(self, row: pd.Series, position_side: int) -> StrategyDecision:
        """Generate entry/exit/hold action from current feature snapshot."""
        required = ["close", "rsi", "ret_5", "regime_ema_fast", "regime_ema_slow", "regime_ret_5"]
        if self.invalid_row(row, required):
            return self.flat("indicator_not_ready")

        dip_th = float(self.params.get("dip_pct_th", 0.035))
        spike_th = float(self.params.get("spike_pct_th", 0.035))
        rsi_low = float(self.params.get("rsi_low", 28.0))
        rsi_high = float(self.params.get("rsi_high", 72.0))
        rebound_exit = float(self.params.get("rebound_exit", 0.02))

        local_dip = float(row["ret_5"]) <= -dip_th
        local_spike = float(row["ret_5"]) >= spike_th
        rsi = float(row["rsi"])

        regime_not_collapse = not (
            float(row["regime_ema_fast"]) < float(row["regime_ema_slow"]) and float(row["regime_ret_5"]) < -0.06
        )
        regime_not_meltup = not (
            float(row["regime_ema_fast"]) > float(row["regime_ema_slow"]) and float(row["regime_ret_5"]) > 0.06
        )

        long_signal = local_dip and rsi <= rsi_low and regime_not_collapse
        short_signal = local_spike and rsi >= rsi_high and regime_not_meltup

        strength = min(2.0, abs(float(row["ret_5"])) / max(dip_th, 1e-9))

        if position_side == FLAT:
            if long_signal:
                return StrategyDecision(LONG, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.65, "dip_buy_long_entry")
            if short_signal:
                return StrategyDecision(SHORT, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.65, "dip_buy_short_entry")
            return self.flat("dip_buy_no_entry")

        if position_side == LONG:
            if float(row["ret_1"]) >= rebound_exit or rsi >= 55.0:
                return StrategyDecision(FLAT, 0.0, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.55, "dip_buy_long_exit")
            return StrategyDecision(LONG, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.4, "dip_buy_hold_long")

        if float(row["ret_1"]) <= -rebound_exit or rsi <= 45.0:
            return StrategyDecision(FLAT, 0.0, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.55, "dip_buy_short_exit")
        return StrategyDecision(SHORT, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.4, "dip_buy_hold_short")
