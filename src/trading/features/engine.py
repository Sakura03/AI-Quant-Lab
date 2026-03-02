from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=max(1, int(period)), adjust=False).mean()


def macd(series: pd.Series, fast: int, slow: int, signal: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / max(1, int(period)), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / max(1, int(period)), adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


def bbands(series: pd.Series, period: int, num_std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger upper/middle/lower bands."""
    p = max(2, int(period))
    mid = series.rolling(p).mean()
    std = series.rolling(p).std(ddof=0)
    upper = mid + float(num_std) * std
    lower = mid - float(num_std) * std
    return upper, mid, lower


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range used by ATR/ADX."""
    prev_close = close.shift(1)
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average true range."""
    tr = true_range(df["high"], df["low"], df["close"])
    return tr.ewm(alpha=1.0 / max(1, int(period)), adjust=False).mean()


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Average Directional Index."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(high, low, close)
    atr_s = tr.ewm(alpha=1.0 / max(1, int(period)), adjust=False).mean().replace(0.0, np.nan)

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / max(1, int(period)), adjust=False).mean() / atr_s
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / max(1, int(period)), adjust=False).mean() / atr_s

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    out = dx.ewm(alpha=1.0 / max(1, int(period)), adjust=False).mean()
    return out.fillna(0.0)


def compute_base_features(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Compute per-timeframe technical feature columns used by strategies."""
    out = df.copy()

    out["ema_fast"] = ema(out["close"], int(params.get("ema_fast", 20)))
    out["ema_slow"] = ema(out["close"], int(params.get("ema_slow", 50)))

    macd_line, signal_line, hist = macd(
        out["close"],
        int(params.get("macd_fast", 12)),
        int(params.get("macd_slow", 26)),
        int(params.get("macd_signal", 9)),
    )
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    out["rsi"] = rsi(out["close"], int(params.get("rsi_period", 14)))

    bb_upper, bb_mid, bb_lower = bbands(
        out["close"],
        int(params.get("bb_period", 20)),
        float(params.get("bb_std", 2.0)),
    )
    out["bb_upper"] = bb_upper
    out["bb_mid"] = bb_mid
    out["bb_lower"] = bb_lower

    out["atr"] = atr(out, int(params.get("atr_period", 14)))
    out["adx"] = adx(out, int(params.get("adx_period", 14)))
    out["ret_1"] = out["close"].pct_change().fillna(0.0)
    out["ret_5"] = out["close"].pct_change(5).fillna(0.0)
    out["vol_20"] = out["ret_1"].rolling(20).std(ddof=0)
    out["zscore_20"] = (out["close"] - out["close"].rolling(20).mean()) / out["close"].rolling(20).std(ddof=0)

    return out


def merge_regime_features(
    signal_features: pd.DataFrame,
    regime_features: pd.DataFrame,
) -> pd.DataFrame:
    """Backward-align regime features to signal close_time without lookahead."""
    keep = [
        "close_time",
        "ema_fast",
        "ema_slow",
        "macd_hist",
        "rsi",
        "bb_mid",
        "atr",
        "adx",
        "ret_1",
        "ret_5",
        "vol_20",
        "zscore_20",
        "close",
    ]
    for col in keep:
        if col not in regime_features.columns:
            raise ValueError(f"Missing regime feature column: {col}")

    r = regime_features[keep].copy().rename(
        columns={
            "ema_fast": "regime_ema_fast",
            "ema_slow": "regime_ema_slow",
            "macd_hist": "regime_macd_hist",
            "rsi": "regime_rsi",
            "bb_mid": "regime_bb_mid",
            "atr": "regime_atr",
            "adx": "regime_adx",
            "ret_1": "regime_ret_1",
            "ret_5": "regime_ret_5",
            "vol_20": "regime_vol_20",
            "zscore_20": "regime_zscore_20",
            "close": "regime_close",
        }
    )

    left = signal_features.sort_values("close_time").reset_index(drop=True)
    right = r.sort_values("close_time").reset_index(drop=True)

    # close_time semantics: only use regime bars that are already closed at signal close_time.
    out = pd.merge_asof(
        left,
        right,
        on="close_time",
        direction="backward",
        allow_exact_matches=True,
    )
    return out


def build_feature_frame(
    signal_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Build final strategy feature frame with signal+regime inputs."""
    signal_features = compute_base_features(signal_df, params)
    regime_features = compute_base_features(regime_df, params)
    merged = merge_regime_features(signal_features, regime_features)
    return merged
