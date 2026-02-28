#!/usr/bin/env python3
"""统计 data 目录下 feather 数据文件的时间覆盖范围。"""

from __future__ import annotations

import argparse
import glob
import os
import os.path as osp
from typing import Tuple

import pandas as pd


FUNDING_SUFFIX = "-funding-rate.feather"


def parse_file_meta(path: str) -> Tuple[str, str]:
    """从文件名中解析 symbol 和 timeframe。"""
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
    """根据时间戳量级推断单位（s/ms/us/ns）。"""
    v = abs(float(value))
    if v >= 1e18:
        return "ns"
    if v >= 1e15:
        return "us"
    if v >= 1e12:
        return "ms"
    return "s"


def normalize_timestamp(series: pd.Series) -> pd.Series:
    """把 timestamp 列标准化为 pandas 时间类型。"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)

    s = series.dropna()
    if s.empty:
        return pd.to_datetime(series)

    first = s.iloc[0]
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        unit = infer_time_unit(first)
        return pd.to_datetime(series, unit=unit)

    return pd.to_datetime(series)


def collect_stats(data_dir: str) -> pd.DataFrame:
    """扫描目录并统计每个 feather 文件的起止时间。"""
    paths = sorted(glob.glob(osp.join(data_dir, "*.feather")))
    rows = []

    for path in paths:
        symbol, timeframe = parse_file_meta(path)

        try:
            df = pd.read_feather(path, columns=["timestamp"])
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "file": osp.basename(path),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "rows": 0,
                    "start_time": pd.NaT,
                    "end_time": pd.NaT,
                    "duration_days": None,
                    "error": str(exc),
                }
            )
            continue

        if "timestamp" not in df.columns or df.empty:
            rows.append(
                {
                    "file": osp.basename(path),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "rows": int(len(df)),
                    "start_time": pd.NaT,
                    "end_time": pd.NaT,
                    "duration_days": 0.0,
                    "error": "missing_or_empty_timestamp",
                }
            )
            continue

        ts = normalize_timestamp(df["timestamp"])
        start_time = ts.min()
        end_time = ts.max()
        duration_days = (end_time - start_time).total_seconds() / 86400.0 if pd.notna(start_time) and pd.notna(end_time) else None

        rows.append(
            {
                "file": osp.basename(path),
                "symbol": symbol,
                "timeframe": timeframe,
                "rows": int(len(df)),
                "start_time": start_time,
                "end_time": end_time,
                "duration_days": round(duration_days, 3) if duration_days is not None else None,
                "error": "",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["symbol", "timeframe", "file"]).reset_index(drop=True)
    return out


def build_symbol_summary(stats_df: pd.DataFrame) -> pd.DataFrame:
    """按交易对聚合统计整体覆盖范围。"""
    if stats_df.empty:
        return pd.DataFrame(columns=["symbol", "datasets", "start_time", "end_time", "coverage_days"])

    ok = stats_df[stats_df["error"] == ""].copy()
    if ok.empty:
        return pd.DataFrame(columns=["symbol", "datasets", "start_time", "end_time", "coverage_days"])

    summary = (
        ok.groupby("symbol", as_index=False)
        .agg(
            datasets=("file", "count"),
            start_time=("start_time", "min"),
            end_time=("end_time", "max"),
        )
    )
    summary["coverage_days"] = ((summary["end_time"] - summary["start_time"]).dt.total_seconds() / 86400.0).round(3)
    return summary.sort_values("symbol").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="统计 data 目录下各 feather 文件的起止时间")
    parser.add_argument("--data-dir", default="data", help="数据目录，默认 data")
    parser.add_argument("--save-csv", default=None, help="可选：将逐文件统计结果保存为 CSV")
    parser.add_argument("--save-summary-csv", default=None, help="可选：将按 symbol 聚合结果保存为 CSV")
    args = parser.parse_args()

    if not osp.isdir(args.data_dir):
        raise FileNotFoundError(f"目录不存在: {args.data_dir}")

    stats_df = collect_stats(args.data_dir)
    summary_df = build_symbol_summary(stats_df)

    print(f"[calc_stat] data_dir={args.data_dir}")
    print(f"[calc_stat] files={len(stats_df)}")

    if stats_df.empty:
        print("没有找到 feather 文件。")
        return

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n=== 逐文件统计 ===")
    print(stats_df.to_string(index=False))

    print("\n=== 按交易对聚合 ===")
    if summary_df.empty:
        print("无可用聚合结果。")
    else:
        print(summary_df.to_string(index=False))

    if args.save_csv:
        stats_df.to_csv(args.save_csv, index=False)
        print(f"\n已保存逐文件统计: {args.save_csv}")

    if args.save_summary_csv:
        summary_df.to_csv(args.save_summary_csv, index=False)
        print(f"已保存聚合统计: {args.save_summary_csv}")


if __name__ == "__main__":
    main()
