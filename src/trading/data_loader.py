from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import os.path as osp

import pandas as pd

from .config import DataConfig


@dataclass
class SymbolFrames:
    """单个交易对的信号周期、执行周期和资金费率数据容器。"""
    signal: pd.DataFrame
    execution: pd.DataFrame
    funding: pd.DataFrame


MarketStore = Dict[str, SymbolFrames]


def symbol_to_filename(symbol: str) -> str:
    """将交易对格式 `BTC/USDT` 转为文件名前缀 `BTC_USDT`。"""
    return symbol.replace("/", "_")


def _read_feather(path: str) -> pd.DataFrame:
    """读取 feather 数据并标准化 timestamp 列为时间类型。"""
    if not osp.isfile(path):
        raise FileNotFoundError(f"找不到数据文件: {path}")
    df = pd.read_feather(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"数据缺少 timestamp 列: {path}")
    df = df.copy()
    if pd.api.types.is_integer_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return df.reset_index(drop=True)


def _clip_with_warmup(df: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp, warmup_bars: int) -> pd.DataFrame:
    """按回测区间截取数据，并额外保留预热 K 线用于指标初始化。"""
    mask = df["timestamp"] <= end_time
    out = df.loc[mask].copy()
    anchor = out[out["timestamp"] < start_time]
    keep_warmup = anchor.tail(warmup_bars)
    main = out[out["timestamp"] >= start_time]
    out = pd.concat([keep_warmup, main], ignore_index=True)
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return out.reset_index(drop=True)


def load_market_store(data_cfg: DataConfig) -> MarketStore:
    """按配置加载所有交易对数据，并返回统一的市场数据字典。"""
    store: MarketStore = {}

    for symbol in data_cfg.symbols:
        base = symbol_to_filename(symbol)

        signal_path = osp.join(data_cfg.data_folder, f"{base}-{data_cfg.timeframe_signal}.feather")
        execution_path = osp.join(data_cfg.data_folder, f"{base}-{data_cfg.timeframe_execution}.feather")
        funding_path = osp.join(data_cfg.data_folder, f"{base}-funding-rate.feather")

        signal_df = _read_feather(signal_path)
        execution_df = _read_feather(execution_path)
        funding_df = _read_feather(funding_path)

        signal_df = _clip_with_warmup(signal_df, data_cfg.start_time, data_cfg.end_time, data_cfg.warmup_bars)
        execution_df = execution_df[(execution_df["timestamp"] >= data_cfg.start_time) & (execution_df["timestamp"] <= data_cfg.end_time)].copy()
        funding_df = funding_df[(funding_df["timestamp"] >= data_cfg.start_time) & (funding_df["timestamp"] <= data_cfg.end_time)].copy()

        if execution_df.empty:
            raise ValueError(f"执行周期数据为空: {symbol}")

        if "fundingRate" not in funding_df.columns:
            funding_df["fundingRate"] = 0.0

        store[symbol] = SymbolFrames(
            signal=signal_df.reset_index(drop=True),
            execution=execution_df.reset_index(drop=True),
            funding=funding_df.reset_index(drop=True),
        )

    return store
