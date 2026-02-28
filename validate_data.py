#!/usr/bin/env python3
"""Validate coverage and continuity of market data files in the data directory.

Outputs per-symbol per-timeframe start/end timestamps and checks for:
- missing files for symbol/timeframe combinations
- duplicate timestamps
- abnormal time gaps (possible missing bars)
- cross-timeframe head/tail truncation within each symbol
"""

from __future__ import annotations

import argparse
import glob
import math
import os.path as osp
import re
from typing import Optional, Tuple

import pandas as pd


FUNDING_SUFFIX = "-funding-rate.feather"


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
        return pd.to_datetime(series).dt.tz_localize(None)

    s = series.dropna()
    if s.empty:
        return pd.to_datetime(series)

    first = s.iloc[0]
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        unit = infer_time_unit(first)
        return pd.to_datetime(series, unit=unit).dt.tz_localize(None)

    return pd.to_datetime(series).dt.tz_localize(None)


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
        # conservative integer estimate; e.g. 3*step => 2 missing bars
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


def enrich_symbol_alignment(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Mark rows whose start/end lag too much behind same-symbol peer datasets."""
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

    ts = normalize_timestamp(df["timestamp"])
    ts = ts.dropna()

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
    if stats_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "datasets",
                "start_time",
                "end_time",
                "coverage_days",
                "files_with_gaps",
                "estimated_missing_bars",
                "files_with_duplicates",
            ]
        )

    okish = stats_df[stats_df["status"].isin(["ok", "warning"])].copy()
    if okish.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "datasets",
                "start_time",
                "end_time",
                "coverage_days",
                "files_with_gaps",
                "estimated_missing_bars",
                "files_with_duplicates",
            ]
        )

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

    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["symbol", "missing_timeframes", "missing_count"]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate data files: per symbol/timeframe coverage and gap checks"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    parser.add_argument(
        "--gap-tolerance",
        type=float,
        default=1.5,
        help="Gap threshold multiplier vs expected step (default: 1.5)",
    )
    parser.add_argument("--only-issues", action="store_true", help="Print only warning/error rows")
    parser.add_argument("--save-csv", default=None, help="Save per-file validation CSV")
    parser.add_argument("--save-summary-csv", default=None, help="Save per-symbol summary CSV")
    parser.add_argument("--save-missing-csv", default=None, help="Save missing file combinations CSV")
    args = parser.parse_args()

    if not osp.isdir(args.data_dir):
        raise FileNotFoundError(f"Directory not found: {args.data_dir}")

    stats_df = collect_stats(args.data_dir, args.gap_tolerance)
    summary_df = build_symbol_summary(stats_df)
    missing_df = build_missing_files(stats_df)

    print(f"[validate_data] data_dir={args.data_dir}")
    print(f"[validate_data] files={len(stats_df)}")

    if stats_df.empty:
        print("No feather files found.")
        return

    warn_count = int((stats_df["status"] == "warning").sum())
    err_count = int((stats_df["status"] == "error").sum())
    print(f"[validate_data] warnings={warn_count} errors={err_count}")

    view_df = stats_df
    if args.only_issues:
        view_df = stats_df[stats_df["status"] != "ok"].copy()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\n=== Per Symbol/Timeframe Coverage & Gap Check ===")
    print(view_df.to_string(index=False))

    print("\n=== Per Symbol Summary ===")
    print(summary_df.to_string(index=False) if not summary_df.empty else "No summary rows.")

    print("\n=== Missing Symbol/Timeframe Files ===")
    print(missing_df.to_string(index=False) if not missing_df.empty else "No missing file combinations.")

    if args.save_csv:
        stats_df.to_csv(args.save_csv, index=False)
        print(f"\nSaved per-file CSV: {args.save_csv}")

    if args.save_summary_csv:
        summary_df.to_csv(args.save_summary_csv, index=False)
        print(f"Saved summary CSV: {args.save_summary_csv}")

    if args.save_missing_csv:
        missing_df.to_csv(args.save_missing_csv, index=False)
        print(f"Saved missing-file CSV: {args.save_missing_csv}")


if __name__ == "__main__":
    main()
