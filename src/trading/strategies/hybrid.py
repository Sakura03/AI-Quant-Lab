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


class HybridRegimeSwitchStrategy(BaseStrategy):
    strategy_id = "hybrid_regime_switch"

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {
            "regime_adx_th": 24.0,
            "dip_th": 0.04,
            "z_th": 1.4,
            "rsi_low": 30.0,
            "rsi_high": 70.0,
            "stop_atr": 2.0,
            "take_atr": 3.2,
            "max_hold": 72,
        }

    @classmethod
    def sample_params(cls, rng: np.random.Generator) -> dict[str, Any]:
        return {
            "regime_adx_th": float(rng.uniform(16.0, 38.0)),
            "dip_th": float(rng.uniform(0.01, 0.10)),
            "z_th": float(rng.uniform(0.6, 3.0)),
            "rsi_low": float(rng.uniform(10.0, 40.0)),
            "rsi_high": float(rng.uniform(60.0, 90.0)),
            "stop_atr": float(rng.uniform(0.8, 4.0)),
            "take_atr": float(rng.uniform(1.2, 6.0)),
            "max_hold": int(rng.integers(12, 200)),
        }

    @classmethod
    def mutate_params(cls, params: dict[str, Any], strength: float, rng: np.random.Generator) -> dict[str, Any]:
        s = max(0.01, float(strength))
        out = dict(params)
        out["regime_adx_th"] = _clip_float(out["regime_adx_th"] + rng.normal(0, 4 * s), 10.0, 50.0)
        out["dip_th"] = _clip_float(out["dip_th"] + rng.normal(0, 0.01 * s), 0.005, 0.16)
        out["z_th"] = _clip_float(out["z_th"] + rng.normal(0, 0.3 * s), 0.3, 5.0)
        out["rsi_low"] = _clip_float(out["rsi_low"] + rng.normal(0, 4 * s), 5.0, 45.0)
        out["rsi_high"] = _clip_float(out["rsi_high"] + rng.normal(0, 4 * s), 55.0, 95.0)
        out["stop_atr"] = _clip_float(out["stop_atr"] + rng.normal(0, 0.4 * s), 0.5, 5.0)
        out["take_atr"] = _clip_float(out["take_atr"] + rng.normal(0, 0.6 * s), 0.8, 8.0)
        out["max_hold"] = _clip_int(out["max_hold"] + rng.normal(0, 20 * s), 8, 260)
        return out

    def decide(self, row: pd.Series, position_side: int) -> StrategyDecision:
        required = [
            "ema_fast",
            "ema_slow",
            "macd_hist",
            "rsi",
            "zscore_20",
            "ret_5",
            "regime_ema_fast",
            "regime_ema_slow",
            "regime_adx",
        ]
        if self.invalid_row(row, required):
            return self.flat("indicator_not_ready")

        regime_adx = float(row["regime_adx"])
        adx_th = float(self.params.get("regime_adx_th", 24.0))
        regime_trend = regime_adx >= adx_th

        if regime_trend:
            long_signal = (
                float(row["ema_fast"]) > float(row["ema_slow"])
                and float(row["macd_hist"]) > 0
                and float(row["regime_ema_fast"]) >= float(row["regime_ema_slow"])
            )
            short_signal = (
                float(row["ema_fast"]) < float(row["ema_slow"])
                and float(row["macd_hist"]) < 0
                and float(row["regime_ema_fast"]) <= float(row["regime_ema_slow"])
            )
            reason_prefix = "hybrid_trend"
            strength = min(2.0, abs(float(row["macd_hist"])) * 5.0 + max(0.0, regime_adx - adx_th) / 20.0)
        else:
            dip_th = float(self.params.get("dip_th", 0.04))
            z_th = float(self.params.get("z_th", 1.4))
            rsi_low = float(self.params.get("rsi_low", 30.0))
            rsi_high = float(self.params.get("rsi_high", 70.0))

            shock_dip = float(row["ret_5"]) <= -dip_th and float(row["rsi"]) <= rsi_low
            shock_spike = float(row["ret_5"]) >= dip_th and float(row["rsi"]) >= rsi_high
            meanrev_dip = float(row["zscore_20"]) <= -z_th
            meanrev_spike = float(row["zscore_20"]) >= z_th
            long_signal = shock_dip or meanrev_dip
            short_signal = shock_spike or meanrev_spike
            reason_prefix = "hybrid_mr"
            strength = min(2.0, max(abs(float(row["zscore_20"])) / max(z_th, 1e-9), abs(float(row["ret_5"])) / max(dip_th, 1e-9)))

        if position_side == FLAT:
            if long_signal:
                return StrategyDecision(LONG, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.68, f"{reason_prefix}_long_entry")
            if short_signal:
                return StrategyDecision(SHORT, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.68, f"{reason_prefix}_short_entry")
            return self.flat(f"{reason_prefix}_no_entry")

        if position_side == LONG:
            if not long_signal:
                return StrategyDecision(FLAT, 0.0, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.55, f"{reason_prefix}_long_exit")
            return StrategyDecision(LONG, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.42, f"{reason_prefix}_hold_long")

        if not short_signal:
            return StrategyDecision(FLAT, 0.0, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.55, f"{reason_prefix}_short_exit")
        return StrategyDecision(SHORT, strength, float(self.params["stop_atr"]), float(self.params["take_atr"]), 0.42, f"{reason_prefix}_hold_short")
