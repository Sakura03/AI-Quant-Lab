from __future__ import annotations

import json
import os
import os.path as osp
from datetime import datetime

import pandas as pd
import yaml
import numpy as np

from trading.config import TradingConfig
from trading.domain.types import BacktestResult, ExperimentResult
from trading.evaluate.regime_eval import regime_breakdown
from trading.report.visualization import (
    write_equity_drawdown_html,
    write_equity_drawdown_png,
    write_symbol_candles_with_trades_html,
)


def build_run_dir(base_dir: str, prefix: str) -> str:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = osp.join(base_dir, f"{prefix}_{run_id}")
    os.makedirs(folder, exist_ok=True)
    return folder


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_backtest_report(result: BacktestResult, config: TradingConfig, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    result.equity_curve.to_csv(osp.join(out_dir, "equity_curve.csv"), index=False)
    result.trades.to_csv(osp.join(out_dir, "trade_list.csv"), index=False)

    with open(osp.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result.metrics.to_dict(), f, ensure_ascii=False, indent=2)

    with open(osp.join(out_dir, "strategy_bundle.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_builtin(result.strategy_bundle), f, allow_unicode=True, sort_keys=False)

    with open(osp.join(out_dir, "resolved_config.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, allow_unicode=True, sort_keys=False)

    write_equity_drawdown_html(result.equity_curve, osp.join(out_dir, "equity_drawdown.html"))
    write_equity_drawdown_png(result.equity_curve, osp.join(out_dir, "equity_drawdown.png"))
    candle_tf = str(result.strategy_bundle.get("signal_tf", config.data.exec_tf))
    write_symbol_candles_with_trades_html(
        result.symbol_execution,
        result.trades,
        osp.join(out_dir, "symbol_candles"),
        candle_timeframe=candle_tf,
    )


def write_optimizer_report(exp: ExperimentResult, config: TradingConfig, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    if not exp.all_trials.empty:
        exp.all_trials.to_csv(osp.join(out_dir, "all_trials.csv"), index=False)
        try:
            exp.all_trials.to_parquet(osp.join(out_dir, "all_trials.parquet"), index=False)
        except Exception:
            pass

    window_df = pd.DataFrame([w.to_dict() for w in exp.window_results])
    window_df.to_csv(osp.join(out_dir, "window_summary.csv"), index=False)

    exp.stitched_equity.to_csv(osp.join(out_dir, "stitched_test_equity.csv"), index=False)
    exp.stitched_trades.to_csv(osp.join(out_dir, "stitched_test_trades.csv"), index=False)

    with open(osp.join(out_dir, "stitched_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(exp.stitched_test_metrics, f, ensure_ascii=False, indent=2)

    with open(osp.join(out_dir, "best_genome.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_builtin(exp.best_genome), f, allow_unicode=True, sort_keys=False)

    with open(osp.join(out_dir, "resolved_config.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, allow_unicode=True, sort_keys=False)

    regime_df = regime_breakdown(exp.stitched_equity, exp.stitched_trades, config.data.exec_tf)
    regime_df.to_csv(osp.join(out_dir, "regime_metrics.csv"), index=False)

    if not window_df.empty:
        diag = window_df[
            [
                "window_id",
                "train_sharpe",
                "train_max_drawdown",
                "val_sharpe",
                "val_max_drawdown",
                "test_sharpe",
                "test_max_drawdown",
                "test_trade_count",
                "train_score",
                "val_score",
                "test_score",
            ]
        ].copy()
        diag["overfit_gap"] = (diag["train_sharpe"] - diag["val_sharpe"]).abs()
        diag.to_csv(osp.join(out_dir, "overfit_diagnostics.csv"), index=False)
