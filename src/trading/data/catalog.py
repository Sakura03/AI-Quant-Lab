from __future__ import annotations

import re

import pandas as pd


def symbol_to_filename(symbol: str) -> str:
    """Convert exchange symbol format (e.g. BTC/USDT) into file-safe name."""
    return symbol.replace("/", "_")


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe text like 15m/4h/1d into minutes."""
    m = re.fullmatch(r"(\d+)([mhd])", timeframe.strip().lower())
    if not m:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    value = int(m.group(1))
    unit = m.group(2)
    scale = {"m": 1, "h": 60, "d": 1440}[unit]
    return value * scale


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Convert timeframe text into pandas Timedelta."""
    return pd.Timedelta(minutes=timeframe_to_minutes(timeframe))
