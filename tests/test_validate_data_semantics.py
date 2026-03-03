import numpy as np
import pandas as pd
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from validate_data import analyze_timeframe_consistency, build_semantics_summary


def _write_pair_open_semantics(root):
    ts = pd.date_range("2025-01-01 00:00:00", periods=20, freq="1min")
    base = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0 + np.arange(len(ts), dtype=float),
            "high": 101.0 + np.arange(len(ts), dtype=float),
            "low": 99.0 + np.arange(len(ts), dtype=float),
            "close": 100.5 + np.arange(len(ts), dtype=float),
            "volume": np.ones(len(ts), dtype=float),
        }
    )
    base.to_feather(root / "BTC_USDT-1m.feather")

    agg = (
        base.assign(parent_ts=base["timestamp"].dt.floor("5min"))
        .groupby("parent_ts", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .rename(columns={"parent_ts": "timestamp"})
    )
    agg.to_feather(root / "BTC_USDT-5m.feather")


def _write_pair_close_semantics(root):
    open_ts = pd.date_range("2025-01-02 00:00:00", periods=20, freq="1min")
    close_ts = open_ts + pd.Timedelta(minutes=1)
    base = pd.DataFrame(
        {
            "timestamp": close_ts,
            "open": 200.0 + np.arange(len(close_ts), dtype=float),
            "high": 201.0 + np.arange(len(close_ts), dtype=float),
            "low": 199.0 + np.arange(len(close_ts), dtype=float),
            "close": 200.5 + np.arange(len(close_ts), dtype=float),
            "volume": np.ones(len(close_ts), dtype=float) * 2.0,
        }
    )
    base.to_feather(root / "ETH_USDT-1m.feather")

    tmp = base.copy()
    tmp["bar_open_time"] = tmp["timestamp"] - pd.Timedelta(minutes=1)
    agg = (
        tmp.assign(parent_open=tmp["bar_open_time"].dt.floor("5min"))
        .sort_values("bar_open_time")
        .groupby("parent_open", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
    )
    agg["timestamp"] = agg["parent_open"] + pd.Timedelta(minutes=5)
    agg = agg[["timestamp", "open", "high", "low", "close", "volume"]]
    agg.to_feather(root / "ETH_USDT-5m.feather")


def test_analyze_timeframe_consistency_infers_open_and_close(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _write_pair_open_semantics(root)
    _write_pair_close_semantics(root)

    out = analyze_timeframe_consistency(
        data_dir=str(root),
        consistency_threshold=0.98,
        semantic_margin=0.02,
        price_rtol=1e-9,
        price_atol=1e-9,
        volume_rtol=1e-9,
        volume_atol=1e-9,
    )

    assert len(out) == 2

    btc = out[out["symbol"] == "BTC_USDT"].iloc[0]
    assert btc["child_tf"] == "1m"
    assert btc["parent_tf"] == "5m"
    assert btc["inferred_semantics"] == "open"
    assert btc["open_match_rate"] == 1.0
    assert btc["close_match_rate"] == 0.0

    eth = out[out["symbol"] == "ETH_USDT"].iloc[0]
    assert eth["inferred_semantics"] == "close"
    assert eth["close_match_rate"] == 1.0
    assert eth["open_match_rate"] == 0.0

    sem = build_semantics_summary(out, semantic_margin=0.02)
    all_row = sem[sem["symbol"] == "__ALL__"].iloc[0]
    assert all_row["inferred_semantics"] == "ambiguous"
