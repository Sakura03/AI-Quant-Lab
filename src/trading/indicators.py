from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """计算指数移动平均线（EMA）。"""
    return series.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """计算 MACD 三元组：快慢线差值、信号线、柱状图。"""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(series: pd.Series, period: int) -> pd.Series:
    """计算相对强弱指标（RSI）。"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


def bbands(series: pd.Series, period: int, num_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """计算布林带上轨、中轨、下轨。"""
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """计算真实波动幅度（TR）。"""
    prev_close = close.shift(1)
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """基于 TR 计算平均真实波动幅度（ATR）。"""
    tr = true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    """计算平均趋向指数（ADX），用于识别趋势强弱。"""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_s = tr.ewm(alpha=1.0 / period, adjust=False).mean().replace(0.0, np.nan)

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_s
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_s

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    out = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return out.fillna(0.0)


def prepare_hybrid_features(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """为混合策略批量生成所需技术指标特征列。"""
    out = df.copy()

    out["ema_fast"] = ema(out["close"], int(params["ema_fast"]))
    out["ema_slow"] = ema(out["close"], int(params["ema_slow"]))
    macd_line, signal_line, hist = macd(
        out["close"],
        int(params["macd_fast"]),
        int(params["macd_slow"]),
        int(params["macd_signal"]),
    )
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    out["rsi"] = rsi(out["close"], int(params["rsi_period"]))
    bb_upper, bb_mid, bb_lower = bbands(out["close"], int(params["bb_period"]), float(params["bb_std"]))
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower

    out["atr"] = atr(out, int(params["atr_period"]))
    out["adx"] = adx(out, int(params["adx_period"]))

    return out
