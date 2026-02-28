from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any

import numpy as np
import pandas as pd

from .indicators import prepare_hybrid_features
from .types import LONG, SHORT, FLAT, Position


@dataclass
class StrategyParams:
    """混合策略参数集合。"""
    ema_fast: int = 20
    ema_slow: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_exit_long: float = 55.0
    rsi_exit_short: float = 45.0

    bb_period: int = 20
    bb_std: float = 2.0

    atr_period: int = 14
    adx_period: int = 14
    adx_threshold: float = 25.0

    stop_atr_mult: float = 2.0
    take_atr_mult: float = 3.0
    max_hold_bars: int = 72

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "StrategyParams":
        """从参数字典构建策略参数；空值时使用默认参数。"""
        if not data:
            return cls()
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """将策略参数序列化为字典。"""
        return asdict(self)


@dataclass
class SignalDecision:
    """单次信号决策结果：目标方向、原因和市场状态。"""
    target_side: int
    reason: str
    regime: str


def sample_strategy_params(rng: np.random.Generator) -> Dict[str, Any]:
    """随机采样一组策略参数，用于 walk-forward 搜索。"""
    ema_fast = int(rng.integers(8, 31))
    ema_slow = int(rng.integers(max(ema_fast + 5, 25), 101))
    macd_fast = int(rng.integers(8, 16))
    macd_slow = int(rng.integers(max(macd_fast + 6, 20), 40))
    macd_signal = int(rng.integers(6, 14))

    params = StrategyParams(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        rsi_period=int(rng.integers(7, 22)),
        rsi_overbought=float(rng.integers(62, 81)),
        rsi_oversold=float(rng.integers(20, 39)),
        rsi_exit_long=float(rng.integers(50, 66)),
        rsi_exit_short=float(rng.integers(35, 51)),
        bb_period=int(rng.integers(14, 35)),
        bb_std=float(rng.uniform(1.6, 2.8)),
        atr_period=int(rng.integers(7, 22)),
        adx_period=int(rng.integers(7, 22)),
        adx_threshold=float(rng.uniform(18.0, 35.0)),
        stop_atr_mult=float(rng.uniform(1.2, 3.0)),
        take_atr_mult=float(rng.uniform(1.5, 4.5)),
        max_hold_bars=int(rng.integers(12, 120)),
    )
    return params.to_dict()


class HybridTrendMeanRevStrategy:
    """趋势跟随 + 均值回归的混合策略实现。"""
    def __init__(self, params: StrategyParams):
        """初始化策略参数。"""
        self.params = params

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """对原始 K 线数据添加策略所需指标列。"""
        return prepare_hybrid_features(df, self.params.to_dict())

    def decide(self, row: pd.Series, position: Position | None) -> SignalDecision:
        """基于当前指标和持仓状态，给出开平仓方向决策。"""
        # 1) 指标可用性检查：
        # 指标尚未就绪（NaN）时不产生新交易信号；若已有持仓则维持原方向。
        if self._invalid_row(row):
            return SignalDecision(target_side=position.side if position else FLAT, reason="indicator_not_ready", regime="none")

        # 2) 市场状态识别：
        # 用 ADX 划分趋势市与震荡市，后续分别走不同决策分支。
        trend_regime = row["adx"] >= self.params.adx_threshold
        regime = "trend" if trend_regime else "mean_reversion"

        # 3) 无持仓分支（入场逻辑）：
        # - 趋势市：EMA 多空排列 + MACD 动量确认 + 价格位置过滤。
        # - 震荡市：RSI 超买超卖 + 布林带边界触达触发反转入场。
        if position is None:
            if trend_regime:
                if row["ema_fast"] > row["ema_slow"] and row["macd_hist"] > 0 and row["close"] >= row["ema_fast"]:
                    return SignalDecision(LONG, "trend_long_entry", regime)
                if row["ema_fast"] < row["ema_slow"] and row["macd_hist"] < 0 and row["close"] <= row["ema_fast"]:
                    return SignalDecision(SHORT, "trend_short_entry", regime)
                return SignalDecision(FLAT, "trend_no_entry", regime)

            if row["rsi"] <= self.params.rsi_oversold and row["close"] <= row["bb_lower"]:
                return SignalDecision(LONG, "meanrev_long_entry", regime)
            if row["rsi"] >= self.params.rsi_overbought and row["close"] >= row["bb_upper"]:
                return SignalDecision(SHORT, "meanrev_short_entry", regime)
            return SignalDecision(FLAT, "meanrev_no_entry", regime)

        # 4) 有持仓通用风控：
        # 持仓超过最大 bar 数时强制平仓，避免无限期持有。
        if position.holding_bars >= self.params.max_hold_bars:
            return SignalDecision(FLAT, "max_holding_bars_exit", regime)

        # 5) 多头持仓管理：
        # - 趋势市：趋势反转迹象（EMA 反转或 MACD 转弱）则离场。
        # - 震荡市：RSI 回归到预设退出阈值则止盈/减仓离场。
        if position.side == LONG:
            if trend_regime:
                if row["ema_fast"] < row["ema_slow"] or row["macd_hist"] < 0:
                    return SignalDecision(FLAT, "trend_long_exit", regime)
            else:
                if row["rsi"] >= self.params.rsi_exit_long:
                    return SignalDecision(FLAT, "meanrev_long_exit", regime)
            return SignalDecision(LONG, "hold_long", regime)

        # 6) 空头持仓管理（与多头对称）：
        # - 趋势市：空头趋势被破坏则离场。
        # - 震荡市：RSI 回落到退出阈值则离场。
        if trend_regime:
            if row["ema_fast"] > row["ema_slow"] or row["macd_hist"] > 0:
                return SignalDecision(FLAT, "trend_short_exit", regime)
        else:
            if row["rsi"] <= self.params.rsi_exit_short:
                return SignalDecision(FLAT, "meanrev_short_exit", regime)

        # 7) 若未触发任何退出条件，则继续持有当前空头仓位。
        return SignalDecision(SHORT, "hold_short", regime)

    @staticmethod
    def _invalid_row(row: pd.Series) -> bool:
        """判断指标是否就绪，防止在 NaN 阶段产生错误信号。"""
        required = ["ema_fast", "ema_slow", "macd_hist", "rsi", "bb_upper", "bb_lower", "atr", "adx"]
        return any(pd.isna(row[c]) for c in required)
