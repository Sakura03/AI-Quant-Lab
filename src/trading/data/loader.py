from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import os.path as osp
import warnings

import pandas as pd

from trading.config import DataConfig
from trading.data.catalog import symbol_to_filename, timeframe_to_timedelta


@dataclass
class SymbolDataBundle:
    signal: pd.DataFrame
    regime: pd.DataFrame
    execution: pd.DataFrame
    funding: pd.DataFrame


class MarketDataLoader:
    """Loads and caches raw symbol/timeframe bars with close-time semantics."""

    def __init__(self, data_cfg: DataConfig):
        self.cfg = data_cfg
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._funding_cache: dict[str, pd.DataFrame] = {}

    def _read_feather(self, path: str) -> pd.DataFrame:
        if not osp.isfile(path):
            raise FileNotFoundError(f"Missing market data file: {path}")
        df = pd.read_feather(path)
        if "timestamp" not in df.columns:
            raise ValueError(f"DataFrame missing timestamp column: {path}")

        out = df.copy()
        if pd.api.types.is_integer_dtype(out["timestamp"]):
            out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ms")
        else:
            out["timestamp"] = pd.to_datetime(out["timestamp"])

        out["close_time"] = out["timestamp"].dt.tz_localize(None)
        out = out.sort_values("close_time").drop_duplicates(subset=["close_time"], keep="last")
        out = out.reset_index(drop=True)
        return out

    def _get_full_bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        key = (symbol, timeframe)
        if key in self._cache:
            return self._cache[key]

        base = symbol_to_filename(symbol)
        path = osp.join(self.cfg.root, f"{base}-{timeframe}.feather")
        bars = self._read_feather(path)
        bars["open_time"] = bars["close_time"] - timeframe_to_timedelta(timeframe)
        bars["timeframe"] = timeframe
        self._cache[key] = bars
        return bars

    def _get_full_funding(self, symbol: str) -> pd.DataFrame:
        if symbol in self._funding_cache:
            return self._funding_cache[symbol]
        base = symbol_to_filename(symbol)
        path = osp.join(self.cfg.root, f"{base}-funding-rate.feather")
        fdf = self._read_feather(path)
        if "fundingRate" not in fdf.columns:
            fdf["fundingRate"] = 0.0
        self._funding_cache[symbol] = fdf[["close_time", "fundingRate"]].copy()
        return self._funding_cache[symbol]

    @staticmethod
    def _slice_with_warmup(
        bars: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        warmup_bars: int,
    ) -> pd.DataFrame:
        capped = bars[bars["close_time"] <= end].copy()
        anchor = capped[capped["close_time"] < start]
        warmup = anchor.tail(max(0, warmup_bars))
        main = capped[capped["close_time"] >= start]
        out = pd.concat([warmup, main], ignore_index=True)
        out = out.sort_values("close_time").drop_duplicates(subset=["close_time"], keep="last")
        return out.reset_index(drop=True)

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        warmup_bars: int,
    ) -> pd.DataFrame:
        full = self._get_full_bars(symbol, timeframe)
        return self._slice_with_warmup(full, start, end, warmup_bars)

    def get_execution_bars(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        bars = self._get_full_bars(symbol, self.cfg.exec_tf)
        out = bars[(bars["close_time"] >= start) & (bars["close_time"] <= end)].copy()
        return out.reset_index(drop=True)

    def get_funding(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        fdf = self._get_full_funding(symbol)
        out = fdf[(fdf["close_time"] >= start) & (fdf["close_time"] <= end)].copy()
        return out.reset_index(drop=True)

    def build_bundle(
        self,
        symbol: str,
        signal_tf: str,
        regime_tf: str | None,
        start: pd.Timestamp,
        end: pd.Timestamp,
        warmup_bars: int,
    ) -> SymbolDataBundle:
        signal_df = self.get_bars(symbol, signal_tf, start, end, warmup_bars)
        if regime_tf is None:
            regime_df = signal_df.copy()
        else:
            regime_df = self.get_bars(symbol, regime_tf, start, end, warmup_bars)
        exec_df = self.get_execution_bars(symbol, start, end)
        funding_df = self.get_funding(symbol, start, end)

        if exec_df.empty:
            raise ValueError(f"Execution frame is empty for {symbol}")

        return SymbolDataBundle(
            signal=signal_df,
            regime=regime_df,
            execution=exec_df,
            funding=funding_df,
        )


def load_market_bundles(
    data_cfg: DataConfig,
    signal_tf: str,
    regime_tf: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup_bars: int,
    loader: MarketDataLoader | None = None,
) -> Dict[str, SymbolDataBundle]:
    data_loader = loader if loader is not None else MarketDataLoader(data_cfg)
    out: Dict[str, SymbolDataBundle] = {}
    skipped: list[str] = []
    for symbol in data_cfg.universe:
        try:
            out[symbol] = data_loader.build_bundle(symbol, signal_tf, regime_tf, start, end, warmup_bars)
        except ValueError as exc:
            # For partial-data universes, skip symbols that have no execution bars in the selected range.
            if "Execution frame is empty" in str(exc):
                skipped.append(symbol)
                continue
            raise

    if skipped:
        warnings.warn(
            "Skipped symbols due to empty execution data in selected range: "
            + ", ".join(skipped),
            RuntimeWarning,
            stacklevel=2,
        )
    if not out:
        raise ValueError(
            "No symbols have valid execution data in selected range. "
            f"Requested symbols: {', '.join(data_cfg.universe)}"
        )
    return out
