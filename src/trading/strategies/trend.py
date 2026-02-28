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


class TrendFollowingStrategy(BaseStrategy):
    strategy_id = "trend_following"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "ema_fast": 20,
            "ema_slow": 55,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "adx_threshold": 24.0,
            "stop_atr": 2.2,
            "take_atr": 3.4,
            "max_hold": 72,
        }

    @classmethod
    def sample_params(cls, rng: np.random.Generator) -> dict[str, Any]:
        ema_fast = int(rng.integers(8, 31))
        ema_slow = int(rng.integers(max(ema_fast + 8, 30), 130))
        macd_fast = int(rng.integers(8, 18))
        macd_slow = int(rng.integers(max(macd_fast + 8, 20), 44))
        return {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "macd_fast": macd_fast,
            "macd_slow": macd_slow,
            "macd_signal": int(rng.integers(6, 14)),
            "adx_period": int(rng.integers(8, 26)),
            "adx_threshold": float(rng.uniform(18.0, 35.0)),
            "stop_atr": float(rng.uniform(1.4, 3.6)),
            "take_atr": float(rng.uniform(2.0, 5.5)),
            "max_hold": int(rng.integers(24, 160)),
        }

    @classmethod
    def mutate_params(cls, params: dict[str, Any], strength: float, rng: np.random.Generator) -> dict[str, Any]:
        out = dict(params)
        s = max(0.01, float(strength))
        out["ema_fast"] = _clip_int(out["ema_fast"] + rng.normal(0, 5 * s), 5, 40)
        out["ema_slow"] = _clip_int(out["ema_slow"] + rng.normal(0, 12 * s), out["ema_fast"] + 6, 180)
        out["macd_fast"] = _clip_int(out["macd_fast"] + rng.normal(0, 2 * s), 5, 20)
        out["macd_slow"] = _clip_int(out["macd_slow"] + rng.normal(0, 4 * s), out["macd_fast"] + 6, 50)
        out["macd_signal"] = _clip_int(out["macd_signal"] + rng.normal(0, 2 * s), 4, 18)
        out["adx_period"] = _clip_int(out["adx_period"] + rng.normal(0, 3 * s), 6, 30)
        out["adx_threshold"] = _clip_float(out["adx_threshold"] + rng.normal(0, 4 * s), 10.0, 45.0)
        out["stop_atr"] = _clip_float(out["stop_atr"] + rng.normal(0, 0.5 * s), 0.8, 5.0)
        out["take_atr"] = _clip_float(out["take_atr"] + rng.normal(0, 0.8 * s), 1.0, 8.0)
        out["max_hold"] = _clip_int(out["max_hold"] + rng.normal(0, 20 * s), 8, 240)
        return out

    def decide(self, row: pd.Series, position_side: int) -> StrategyDecision:
        required = [
            "ema_fast",
            "ema_slow",
            "macd_hist",
            "adx",
            "regime_ema_fast",
            "regime_ema_slow",
            "regime_adx",
        ]
        if self.invalid_row(row, required):
            return self.flat("indicator_not_ready")

        adx_th = float(self.params.get("adx_threshold", 24.0))
        trend_ok = row["adx"] >= adx_th and row["regime_adx"] >= adx_th * 0.7

        long_signal = (
            trend_ok
            and row["ema_fast"] > row["ema_slow"]
            and row["macd_hist"] > 0
            and row["regime_ema_fast"] >= row["regime_ema_slow"]
        )
        short_signal = (
            trend_ok
            and row["ema_fast"] < row["ema_slow"]
            and row["macd_hist"] < 0
            and row["regime_ema_fast"] <= row["regime_ema_slow"]
        )

        signal_strength = min(2.0, abs(float(row["macd_hist"])) * 6.0 + max(0.0, float(row["adx"]) - adx_th) / 20.0)

        if position_side == FLAT:
            if long_signal:
                return StrategyDecision(LONG, signal_strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.7, "trend_long_entry")
            if short_signal:
                return StrategyDecision(SHORT, signal_strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.7, "trend_short_entry")
            return self.flat("trend_no_entry")

        if position_side == LONG:
            if not long_signal:
                return StrategyDecision(FLAT, 0.0, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.6, "trend_long_exit")
            return StrategyDecision(LONG, signal_strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.4, "trend_hold_long")

        if not short_signal:
            return StrategyDecision(FLAT, 0.0, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.6, "trend_short_exit")
        return StrategyDecision(SHORT, signal_strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.4, "trend_hold_short")
