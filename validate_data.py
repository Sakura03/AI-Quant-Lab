#!/usr/bin/env python3
"""Validate market-data coverage, timestamp semantics, and timeframe consistency."""

from __future__ import annotations

import argparse
import glob
import math
import os.path as osp
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd


FUNDING_SUFFIX = "-funding-rate.feather"
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def parse_file_meta(path: str) -> Tuple[str, str]:
    """Parse symbol and timeframe from filename."""
    filename = osp.basename(path)
    if filename.endswith(FUNDING_SUFFIX):
        symbol = filename[: -len(FUNDING_SUFFIX)]
        return symbol, "funding-rate"

    stem = filename[:-len(".feather")]
    if "-" not in stem:
        return stem, "unknown"
    symbol, timeframe = stem.rsplit("-", 1)
    return symbol, timeframe


def infer_time_unit(value: int | float) -> str:
    """Infer epoch unit from magnitude."""
    v = abs(float(value))
    if v >= 1e18:
        return "ns"
    if v >= 1e15:
        return "us"
    if v >= 1e12:
        return "ms"
    return "s"


def normalize_timestamp(series: pd.Series) -> pd.Series:
    """Normalize timestamp column into timezone-naive pandas datetime."""
    if pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series)
    else:
        s = series.dropna()
        if s.empty:
            dt = pd.to_datetime(series)
        else:
            first = s.iloc[0]
            if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
                unit = infer_time_unit(first)
                dt = pd.to_datetime(series, unit=unit)
            else:
                dt = pd.to_datetime(series)

    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt


def parse_timeframe_delta(timeframe: str) -> Optional[pd.Timedelta]:
    """Convert timeframe text like 1m/4h/1d into timedelta."""
    if timeframe == "funding-rate":
        return None
    m = re.fullmatch(r"(\d+)([mhd])", timeframe.lower())
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2)
    minutes = value * {"m": 1, "h": 60, "d": 1440}[unit]
    return pd.Timedelta(minutes=minutes)


def timeframe_minutes(timeframe: str) -> Optional[int]:
    """Parse timeframe to minutes."""
    td = parse_timeframe_delta(timeframe)
    if td is None:
        return None
    return int(td.total_seconds() // 60)


def infer_expected_delta(timeframe: str, diffs: pd.Series) -> Optional[pd.Timedelta]:
    """Use nominal timeframe first; fallback to median observed step."""
    nominal = parse_timeframe_delta(timeframe)
    if nominal is not None:
        return nominal
    if diffs.empty:
        return None
    med = diffs.median()
    if pd.isna(med) or med <= pd.Timedelta(0):
        return None
    return med


def estimate_missing_bars(diffs: pd.Series, expected: pd.Timedelta, gap_tolerance: float) -> tuple[int, int]:
    """Estimate number of gap segments and missing bars from observed diffs."""
    if expected is None or expected <= pd.Timedelta(0) or diffs.empty:
        return 0, 0

    threshold = expected * float(gap_tolerance)
    gap_count = 0
    missing = 0

    for delta in diffs:
        if delta <= threshold:
            continue
        gap_count += 1
        ratio = delta / expected
        miss = max(int(math.floor(float(ratio))) - 1, 1)
        missing += miss

    return gap_count, missing


def append_issue(issues: str, issue: str) -> str:
    """Append issue tag into comma-separated issue string."""
    items = [x for x in str(issues).split(",") if x]
    if issue not in items:
        items.append(issue)
    return ",".join(items)


def parse_timedelta(value: object) -> Optional[pd.Timedelta]:
    """Parse timedelta value from string/object; return None on failure."""
    if value is None:
        return None
    if isinstance(value, pd.Timedelta):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        td = pd.to_timedelta(text)
    except Exception:  # noqa: BLE001
        return None
    return td if td > pd.Timedelta(0) else None


def timedelta_to_floor_freq(delta: pd.Timedelta) -> str:
    """Convert timedelta to floor() frequency string."""
    minutes = int(delta.total_seconds() // 60)
    if minutes <= 0:
        raise ValueError(f"Invalid non-positive timeframe delta: {delta}")
    if minutes % 1440 == 0:
        return f"{minutes // 1440}D"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}min"


def enrich_symbol_alignment(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Mark rows whose end lags too much behind same-symbol peer datasets."""
    if stats_df.empty:
        return stats_df

    out = stats_df.copy()
    out["symbol_start_lag"] = pd.Timedelta(0)
    out["symbol_end_lag"] = pd.Timedelta(0)

    valid_mask = out["status"].isin(["ok", "warning"])
    valid_df = out[valid_mask]
    if valid_df.empty:
        return out

    floor_threshold = pd.Timedelta(hours=12)
    for symbol, grp in valid_df.groupby("symbol"):
        symbol_start = grp["start_time"].min()
        symbol_end = grp["end_time"].max()

        for idx, row in grp.iterrows():
            expected = parse_timedelta(row.get("expected_step"))
            dynamic_threshold = expected * 3 if expected is not None else pd.Timedelta(0)
            threshold = max(floor_threshold, dynamic_threshold)

            start_lag = row["start_time"] - symbol_start
            end_lag = symbol_end - row["end_time"]
            out.at[idx, "symbol_start_lag"] = start_lag
            out.at[idx, "symbol_end_lag"] = end_lag

            issues = str(row.get("issues", "") or "")
            status = row.get("status", "ok")
            if end_lag > threshold:
                issues = append_issue(issues, "end_earlier_than_symbol_max")
                status = "warning"
            out.at[idx, "issues"] = issues
            out.at[idx, "status"] = status

    return out


def analyze_file(path: str, gap_tolerance: float) -> dict:
    """Analyze one feather file and return validation row."""
    symbol, timeframe = parse_file_meta(path)
    filename = osp.basename(path)

    base = {
        "file": filename,
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": 0,
        "start_time": pd.NaT,
        "end_time": pd.NaT,
        "expected_step": "",
        "min_step": "",
        "median_step": "",
        "max_step": "",
        "duplicate_timestamps": 0,
        "gap_count": 0,
        "estimated_missing_bars": 0,
        "symbol_start_lag": pd.Timedelta(0),
        "symbol_end_lag": pd.Timedelta(0),
        "status": "error",
        "issues": "",
        "error": "",
    }

    try:
        df = pd.read_feather(path, columns=["timestamp"])
    except Exception as exc:  # noqa: BLE001
        row = dict(base)
        row["error"] = str(exc)
        row["issues"] = "read_error"
        return row

    if "timestamp" not in df.columns:
        row = dict(base)
        row["issues"] = "missing_timestamp_column"
        return row

    ts = normalize_timestamp(df["timestamp"]).dropna()
    row = dict(base)
    row["rows"] = int(len(ts))

    if ts.empty:
        row["issues"] = "empty_timestamp"
        return row

    sorted_ts = ts.sort_values().reset_index(drop=True)
    row["start_time"] = sorted_ts.iloc[0]
    row["end_time"] = sorted_ts.iloc[-1]

    duplicates = int(sorted_ts.duplicated().sum())
    row["duplicate_timestamps"] = duplicates

    uniq = sorted_ts.drop_duplicates().reset_index(drop=True)
    diffs = uniq.diff().dropna()

    expected = infer_expected_delta(timeframe, diffs)
    if expected is not None:
        row["expected_step"] = str(expected)

    if not diffs.empty:
        row["min_step"] = str(diffs.min())
        row["median_step"] = str(diffs.median())
        row["max_step"] = str(diffs.max())

    gap_count, missing = estimate_missing_bars(diffs, expected, gap_tolerance)
    row["gap_count"] = int(gap_count)
    row["estimated_missing_bars"] = int(missing)

    issues = []
    if duplicates > 0:
        issues.append("duplicate_timestamps")
    if gap_count > 0:
        issues.append("time_gaps")
    if timeframe == "unknown":
        issues.append("unknown_timeframe")

    if issues:
        row["status"] = "warning"
        row["issues"] = ",".join(issues)
    else:
        row["status"] = "ok"
        row["issues"] = ""

    return row


def collect_stats(data_dir: str, gap_tolerance: float) -> pd.DataFrame:
    """Scan all feather files in data_dir and build validation table."""
    paths = sorted(glob.glob(osp.join(data_dir, "*.feather")))
    rows = [analyze_file(path, gap_tolerance) for path in paths]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = enrich_symbol_alignment(out)
    return out.sort_values(["symbol", "timeframe", "file"]).reset_index(drop=True)


def build_symbol_summary(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-symbol summary stats."""
    columns = [
        "symbol",
        "datasets",
        "start_time",
        "end_time",
        "coverage_days",
        "files_with_gaps",
        "estimated_missing_bars",
        "files_with_duplicates",
    ]
    if stats_df.empty:
        return pd.DataFrame(columns=columns)

    okish = stats_df[stats_df["status"].isin(["ok", "warning"])].copy()
    if okish.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        okish.groupby("symbol", as_index=False)
        .agg(
            datasets=("file", "count"),
            start_time=("start_time", "min"),
            end_time=("end_time", "max"),
            files_with_gaps=("gap_count", lambda x: int((x > 0).sum())),
            estimated_missing_bars=("estimated_missing_bars", "sum"),
            files_with_duplicates=("duplicate_timestamps", lambda x: int((x > 0).sum())),
        )
    )
    summary["coverage_days"] = ((summary["end_time"] - summary["start_time"]).dt.total_seconds() / 86400.0).round(3)
    return summary.sort_values("symbol").reset_index(drop=True)


def build_missing_files(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Detect symbol/timeframe file-level missing combinations."""
    if stats_df.empty:
        return pd.DataFrame(columns=["symbol", "missing_timeframes", "missing_count"])

    symbols = sorted(stats_df["symbol"].unique().tolist())
    timeframes = sorted(stats_df["timeframe"].unique().tolist())

    present = set(zip(stats_df["symbol"], stats_df["timeframe"]))
    rows = []
    for symbol in symbols:
        missing = [tf for tf in timeframes if (symbol, tf) not in present]
        if not missing:
            continue
        rows.append(
            {
                "symbol": symbol,
                "missing_timeframes": ",".join(missing),
                "missing_count": len(missing),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["symbol", "missing_timeframes", "missing_count"])
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def _read_ohlcv_frame(path: str) -> pd.DataFrame:
    """Read one OHLCV feather frame and normalize timestamp ordering."""
    cols = ["timestamp"] + OHLCV_COLS
    df = pd.read_feather(path, columns=cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    out = df.copy()
    out["timestamp"] = normalize_timestamp(out["timestamp"])
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")

    for col in OHLCV_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=OHLCV_COLS).reset_index(drop=True)
    return out[["timestamp"] + OHLCV_COLS]


def _select_parent_child_pairs(timeframes: list[str]) -> list[tuple[str, str]]:
    """Select one closest divisor-child pair for each parent timeframe."""
    tfs = []
    for tf in timeframes:
        mins = timeframe_minutes(tf)
        if mins is None:
            continue
        tfs.append((tf, mins))
    tfs.sort(key=lambda x: x[1])
    if len(tfs) < 2:
        return []

    pairs: list[tuple[str, str]] = []
    for idx in range(1, len(tfs)):
        parent_tf, parent_mins = tfs[idx]
        candidates = [x for x in tfs[:idx] if parent_mins % x[1] == 0]
        if not candidates:
            continue
        child_tf = max(candidates, key=lambda x: x[1])[0]
        pairs.append((child_tf, parent_tf))
    return pairs


def _aggregate_child_to_parent(
    child_df: pd.DataFrame,
    child_delta: pd.Timedelta,
    parent_delta: pd.Timedelta,
    semantics: str,
) -> pd.DataFrame:
    """Aggregate child OHLCV rows into parent bars under one timestamp semantics."""
    c = child_df.copy()
    if semantics == "open":
        c["_bar_open_time"] = c["timestamp"]
    else:
        c["_bar_open_time"] = c["timestamp"] - child_delta

    c["_parent_open_time"] = c["_bar_open_time"].dt.floor(timedelta_to_floor_freq(parent_delta))
    c = c.sort_values("_bar_open_time")
    agg = (
        c.groupby("_parent_open_time", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            child_count=("timestamp", "count"),
        )
        .reset_index(drop=True)
    )
    if semantics == "open":
        agg["timestamp"] = agg["_parent_open_time"]
    else:
        agg["timestamp"] = agg["_parent_open_time"] + parent_delta

    return agg[["timestamp"] + OHLCV_COLS + ["child_count"]]


def _pair_match_stats(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    child_delta: pd.Timedelta,
    parent_delta: pd.Timedelta,
    semantics: str,
    price_rtol: float,
    price_atol: float,
    volume_rtol: float,
    volume_atol: float,
) -> dict:
    """Evaluate one (child -> parent) aggregation under one semantics."""
    ratio = int(parent_delta / child_delta)
    agg = _aggregate_child_to_parent(child_df, child_delta, parent_delta, semantics)
    merged = parent_df.merge(agg, on="timestamp", how="inner", suffixes=("_parent", "_agg"))

    overlap_rows = int(len(merged))
    if overlap_rows == 0:
        return {
            "overlap_rows": 0,
            "full_rows": 0,
            "match_rows": 0,
            "match_rate": float("nan"),
            "first_mismatch": pd.NaT,
        }

    full = merged[merged["child_count"] == ratio].copy()
    full_rows = int(len(full))
    if full_rows == 0:
        return {
            "overlap_rows": overlap_rows,
            "full_rows": 0,
            "match_rows": 0,
            "match_rate": float("nan"),
            "first_mismatch": pd.NaT,
        }

    price_match = np.isclose(full["open_parent"], full["open_agg"], rtol=price_rtol, atol=price_atol)
    price_match &= np.isclose(full["high_parent"], full["high_agg"], rtol=price_rtol, atol=price_atol)
    price_match &= np.isclose(full["low_parent"], full["low_agg"], rtol=price_rtol, atol=price_atol)
    price_match &= np.isclose(full["close_parent"], full["close_agg"], rtol=price_rtol, atol=price_atol)
    volume_match = np.isclose(full["volume_parent"], full["volume_agg"], rtol=volume_rtol, atol=volume_atol)
    all_match = price_match & volume_match

    match_rows = int(all_match.sum())
    first_mismatch = pd.NaT
    if match_rows < full_rows:
        first_mismatch = full.loc[~all_match, "timestamp"].iloc[0]

    return {
        "overlap_rows": overlap_rows,
        "full_rows": full_rows,
        "match_rows": match_rows,
        "match_rate": float(match_rows / full_rows),
        "first_mismatch": first_mismatch,
    }


def analyze_timeframe_consistency(
    data_dir: str,
    consistency_threshold: float,
    semantic_margin: float,
    price_rtol: float,
    price_atol: float,
    volume_rtol: float,
    volume_atol: float,
) -> pd.DataFrame:
    """Validate cross-timeframe OHLCV consistency and infer timestamp semantics."""
    paths = sorted(glob.glob(osp.join(data_dir, "*.feather")))
    symbol_tf_paths: dict[str, dict[str, str]] = {}
    for path in paths:
        symbol, timeframe = parse_file_meta(path)
        if parse_timeframe_delta(timeframe) is None:
            continue
        symbol_tf_paths.setdefault(symbol, {})[timeframe] = path

    rows: list[dict] = []

    for symbol in sorted(symbol_tf_paths):
        tf_paths = symbol_tf_paths[symbol]
        pairs = _select_parent_child_pairs(list(tf_paths.keys()))
        if not pairs:
            continue

        cache: dict[str, pd.DataFrame] = {}

        for child_tf, parent_tf in pairs:
            child_delta = parse_timeframe_delta(child_tf)
            parent_delta = parse_timeframe_delta(parent_tf)
            ratio = int(parent_delta / child_delta) if child_delta and parent_delta else 0

            try:
                child_df = cache.get(child_tf)
                if child_df is None:
                    child_df = _read_ohlcv_frame(tf_paths[child_tf])
                    cache[child_tf] = child_df
                parent_df = cache.get(parent_tf)
                if parent_df is None:
                    parent_df = _read_ohlcv_frame(tf_paths[parent_tf])
                    cache[parent_tf] = parent_df
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "symbol": symbol,
                        "child_tf": child_tf,
                        "parent_tf": parent_tf,
                        "ratio": ratio,
                        "parent_rows": 0,
                        "open_overlap_rows": 0,
                        "open_full_rows": 0,
                        "open_match_rate": np.nan,
                        "open_first_mismatch": pd.NaT,
                        "close_overlap_rows": 0,
                        "close_full_rows": 0,
                        "close_match_rate": np.nan,
                        "close_first_mismatch": pd.NaT,
                        "inferred_semantics": "unknown",
                        "best_match_rate": np.nan,
                        "status": "error",
                        "issues": "read_ohlcv_error",
                        "error": str(exc),
                    }
                )
                continue

            open_stats = _pair_match_stats(
                child_df=child_df,
                parent_df=parent_df,
                child_delta=child_delta,
                parent_delta=parent_delta,
                semantics="open",
                price_rtol=price_rtol,
                price_atol=price_atol,
                volume_rtol=volume_rtol,
                volume_atol=volume_atol,
            )
            close_stats = _pair_match_stats(
                child_df=child_df,
                parent_df=parent_df,
                child_delta=child_delta,
                parent_delta=parent_delta,
                semantics="close",
                price_rtol=price_rtol,
                price_atol=price_atol,
                volume_rtol=volume_rtol,
                volume_atol=volume_atol,
            )

            open_rate = open_stats["match_rate"]
            close_rate = close_stats["match_rate"]
            valid_rates = [x for x in [open_rate, close_rate] if np.isfinite(x)]
            best_rate = max(valid_rates) if valid_rates else np.nan

            if not np.isfinite(open_rate) and not np.isfinite(close_rate):
                inferred = "unknown"
            elif not np.isfinite(close_rate):
                inferred = "open"
            elif not np.isfinite(open_rate):
                inferred = "close"
            elif open_rate > close_rate + semantic_margin:
                inferred = "open"
            elif close_rate > open_rate + semantic_margin:
                inferred = "close"
            else:
                inferred = "ambiguous"

            issues = []
            if open_stats["overlap_rows"] == 0 and close_stats["overlap_rows"] == 0:
                issues.append("no_timestamp_overlap")
            if open_stats["full_rows"] == 0 and close_stats["full_rows"] == 0:
                issues.append("no_complete_parent_bars")
            if inferred == "ambiguous":
                issues.append("ambiguous_semantics")
            if np.isfinite(best_rate) and best_rate < consistency_threshold:
                issues.append("low_ohlcv_match")

            status = "warning" if issues else "ok"

            rows.append(
                {
                    "symbol": symbol,
                    "child_tf": child_tf,
                    "parent_tf": parent_tf,
                    "ratio": ratio,
                    "parent_rows": int(len(parent_df)),
                    "open_overlap_rows": open_stats["overlap_rows"],
                    "open_full_rows": open_stats["full_rows"],
                    "open_match_rate": open_rate,
                    "open_first_mismatch": open_stats["first_mismatch"],
                    "close_overlap_rows": close_stats["overlap_rows"],
                    "close_full_rows": close_stats["full_rows"],
                    "close_match_rate": close_rate,
                    "close_first_mismatch": close_stats["first_mismatch"],
                    "inferred_semantics": inferred,
                    "best_match_rate": best_rate,
                    "status": status,
                    "issues": ",".join(issues),
                    "error": "",
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "child_tf",
                "parent_tf",
                "ratio",
                "parent_rows",
                "open_overlap_rows",
                "open_full_rows",
                "open_match_rate",
                "open_first_mismatch",
                "close_overlap_rows",
                "close_full_rows",
                "close_match_rate",
                "close_first_mismatch",
                "inferred_semantics",
                "best_match_rate",
                "status",
                "issues",
                "error",
            ]
        )

    out = pd.DataFrame(rows)
    out["child_minutes"] = out["child_tf"].map(timeframe_minutes)
    out["parent_minutes"] = out["parent_tf"].map(timeframe_minutes)
    out = out.sort_values(["symbol", "parent_minutes", "child_minutes", "parent_tf", "child_tf"]).reset_index(drop=True)
    out = out.drop(columns=["child_minutes", "parent_minutes"])
    return out


def _weighted_mean_rate(df: pd.DataFrame, rate_col: str, weight_col: str) -> tuple[float, int]:
    """Compute weighted mean over finite rates with positive weights."""
    use = df[np.isfinite(df[rate_col]) & (df[weight_col] > 0)].copy()
    if use.empty:
        return float("nan"), 0
    weights = use[weight_col].astype(float).values
    rates = use[rate_col].astype(float).values
    return float(np.average(rates, weights=weights)), int(weights.sum())


def build_semantics_summary(consistency_df: pd.DataFrame, semantic_margin: float) -> pd.DataFrame:
    """Aggregate per-pair inference into per-symbol semantics summary."""
    columns = [
        "symbol",
        "comparisons",
        "open_weighted_rate",
        "open_weight",
        "close_weighted_rate",
        "close_weight",
        "inferred_semantics",
        "status",
        "issues",
    ]
    if consistency_df.empty:
        return pd.DataFrame(columns=columns)

    valid = consistency_df[consistency_df["status"] != "error"].copy()
    if valid.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for symbol, grp in valid.groupby("symbol"):
        open_rate, open_weight = _weighted_mean_rate(grp, "open_match_rate", "open_full_rows")
        close_rate, close_weight = _weighted_mean_rate(grp, "close_match_rate", "close_full_rows")

        if not np.isfinite(open_rate) and not np.isfinite(close_rate):
            inferred = "unknown"
        elif not np.isfinite(close_rate):
            inferred = "open"
        elif not np.isfinite(open_rate):
            inferred = "close"
        elif open_rate > close_rate + semantic_margin:
            inferred = "open"
        elif close_rate > open_rate + semantic_margin:
            inferred = "close"
        else:
            inferred = "ambiguous"

        issues = []
        if inferred in {"unknown", "ambiguous"}:
            issues.append("insufficient_or_ambiguous_evidence")
        status = "warning" if issues else "ok"

        rows.append(
            {
                "symbol": symbol,
                "comparisons": int(len(grp)),
                "open_weighted_rate": open_rate,
                "open_weight": open_weight,
                "close_weighted_rate": close_rate,
                "close_weight": close_weight,
                "inferred_semantics": inferred,
                "status": status,
                "issues": ",".join(issues),
            }
        )

    out = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

    all_open_rate, all_open_weight = _weighted_mean_rate(valid, "open_match_rate", "open_full_rows")
    all_close_rate, all_close_weight = _weighted_mean_rate(valid, "close_match_rate", "close_full_rows")

    if not np.isfinite(all_open_rate) and not np.isfinite(all_close_rate):
        all_inferred = "unknown"
    elif not np.isfinite(all_close_rate):
        all_inferred = "open"
    elif not np.isfinite(all_open_rate):
        all_inferred = "close"
    elif all_open_rate > all_close_rate + semantic_margin:
        all_inferred = "open"
    elif all_close_rate > all_open_rate + semantic_margin:
        all_inferred = "close"
    else:
        all_inferred = "ambiguous"

    all_issues = "insufficient_or_ambiguous_evidence" if all_inferred in {"unknown", "ambiguous"} else ""
    all_status = "warning" if all_issues else "ok"
    all_row = pd.DataFrame(
        [
            {
                "symbol": "__ALL__",
                "comparisons": int(len(valid)),
                "open_weighted_rate": all_open_rate,
                "open_weight": all_open_weight,
                "close_weighted_rate": all_close_rate,
                "close_weight": all_close_weight,
                "inferred_semantics": all_inferred,
                "status": all_status,
                "issues": all_issues,
            }
        ]
    )
    return pd.concat([out, all_row], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Validate data files: coverage/gaps, timestamp semantics, and cross-timeframe consistency"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument(
        "--gap-tolerance",
        type=float,
        default=1.5,
        help="Gap threshold multiplier vs expected step (default: 1.5)",
    )
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.98,
        help="Warn if best OHLCV consistency is below this rate (default: 0.98)",
    )
    parser.add_argument(
        "--semantic-margin",
        type=float,
        default=0.02,
        help="Minimum match-rate lead required to infer open/close semantics (default: 0.02)",
    )
    parser.add_argument("--price-rtol", type=float, default=1e-6, help="Relative tolerance for OHLC match")
    parser.add_argument("--price-atol", type=float, default=1e-8, help="Absolute tolerance for OHLC match")
    parser.add_argument("--volume-rtol", type=float, default=1e-6, help="Relative tolerance for volume match")
    parser.add_argument("--volume-atol", type=float, default=1e-8, help="Absolute tolerance for volume match")
    parser.add_argument("--expect-semantics", choices=["open", "close"], default=None, help="Expected global semantics")
    parser.add_argument("--only-issues", action="store_true", help="Print only warning/error rows")
    parser.add_argument("--save-csv", default=None, help="Save per-file validation CSV")
    parser.add_argument("--save-summary-csv", default=None, help="Save per-symbol summary CSV")
    parser.add_argument("--save-missing-csv", default=None, help="Save missing file combinations CSV")
    parser.add_argument("--save-consistency-csv", default=None, help="Save cross-timeframe consistency CSV")
    parser.add_argument("--save-semantics-csv", default=None, help="Save inferred semantics CSV")
    args = parser.parse_args()

    if not osp.isdir(args.data_dir):
        raise FileNotFoundError(f"Directory not found: {args.data_dir}")

    stats_df = collect_stats(args.data_dir, args.gap_tolerance)
    summary_df = build_symbol_summary(stats_df)
    missing_df = build_missing_files(stats_df)
    consistency_df = analyze_timeframe_consistency(
        data_dir=args.data_dir,
        consistency_threshold=args.consistency_threshold,
        semantic_margin=args.semantic_margin,
        price_rtol=args.price_rtol,
        price_atol=args.price_atol,
        volume_rtol=args.volume_rtol,
        volume_atol=args.volume_atol,
    )
    semantics_df = build_semantics_summary(consistency_df, args.semantic_margin)

    print(f"[validate_data] data_dir={args.data_dir}")
    print(f"[validate_data] files={len(stats_df)}")

    if stats_df.empty:
        print("No feather files found.")
        return

    file_warn_count = int((stats_df["status"] == "warning").sum())
    file_err_count = int((stats_df["status"] == "error").sum())
    consistency_warn_count = int((consistency_df["status"] == "warning").sum()) if not consistency_df.empty else 0
    consistency_err_count = int((consistency_df["status"] == "error").sum()) if not consistency_df.empty else 0
    semantics_warn_count = int((semantics_df["status"] == "warning").sum()) if not semantics_df.empty else 0

    print(
        "[validate_data] file_warnings="
        f"{file_warn_count} file_errors={file_err_count} "
        f"consistency_warnings={consistency_warn_count} consistency_errors={consistency_err_count} "
        f"semantics_warnings={semantics_warn_count}"
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)

    file_view = stats_df if not args.only_issues else stats_df[stats_df["status"] != "ok"].copy()
    print("\n=== Per Symbol/Timeframe Coverage & Gap Check ===")
    print(file_view.to_string(index=False))

    print("\n=== Per Symbol Summary ===")
    print(summary_df.to_string(index=False) if not summary_df.empty else "No summary rows.")

    print("\n=== Missing Symbol/Timeframe Files ===")
    print(missing_df.to_string(index=False) if not missing_df.empty else "No missing file combinations.")

    consistency_view = consistency_df if not args.only_issues else consistency_df[consistency_df["status"] != "ok"].copy()
    print("\n=== Cross-Timeframe OHLCV Consistency ===")
    print(consistency_view.to_string(index=False) if not consistency_view.empty else "No consistency rows.")

    semantics_view = semantics_df if not args.only_issues else semantics_df[semantics_df["status"] != "ok"].copy()
    print("\n=== Inferred Timestamp Semantics ===")
    print(semantics_view.to_string(index=False) if not semantics_view.empty else "No semantics rows.")

    if args.expect_semantics and not semantics_df.empty:
        global_rows = semantics_df[semantics_df["symbol"] == "__ALL__"]
        if not global_rows.empty:
            inferred = str(global_rows.iloc[0]["inferred_semantics"])
            if inferred != args.expect_semantics:
                print(
                    "[validate_data] EXPECTATION_MISMATCH "
                    f"expected={args.expect_semantics} inferred={inferred}"
                )

    if args.save_csv:
        stats_df.to_csv(args.save_csv, index=False)
        print(f"\nSaved per-file CSV: {args.save_csv}")

    if args.save_summary_csv:
        summary_df.to_csv(args.save_summary_csv, index=False)
        print(f"Saved summary CSV: {args.save_summary_csv}")

    if args.save_missing_csv:
        missing_df.to_csv(args.save_missing_csv, index=False)
        print(f"Saved missing-file CSV: {args.save_missing_csv}")

    if args.save_consistency_csv:
        consistency_df.to_csv(args.save_consistency_csv, index=False)
        print(f"Saved consistency CSV: {args.save_consistency_csv}")

    if args.save_semantics_csv:
        semantics_df.to_csv(args.save_semantics_csv, index=False)
        print(f"Saved semantics CSV: {args.save_semantics_csv}")


if __name__ == "__main__":
    main()
