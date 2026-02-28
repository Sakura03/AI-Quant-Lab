from __future__ import annotations

import json
import os
import os.path as osp
from datetime import datetime

import yaml

from .config import TradingConfig
from .types import BacktestResult
from .visualization import plot_equity_curve, plot_all_symbols


def build_run_folder(base_folder: str, prefix: str) -> str:
    """按时间戳创建本次运行输出目录并返回路径。"""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = osp.join(base_folder, f"{prefix}_{run_id}")
    os.makedirs(folder, exist_ok=True)
    return folder


def write_backtest_report(result: BacktestResult, config: TradingConfig, save_folder: str):
    """将回测结果写出为 CSV/JSON/YAML/Markdown 与图表文件。"""
    os.makedirs(save_folder, exist_ok=True)

    equity_path = osp.join(save_folder, "equity_curve.csv")
    trades_path = osp.join(save_folder, "trade_list.csv")
    metrics_path = osp.join(save_folder, "metrics.json")
    params_path = osp.join(save_folder, "strategy_params.yml")
    summary_path = osp.join(save_folder, "summary.md")

    result.equity_curve.to_csv(equity_path, index=False)
    result.trades.to_csv(trades_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result.metrics.to_dict(), f, ensure_ascii=False, indent=2)

    with open(params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(result.strategy_params, f, allow_unicode=True, sort_keys=False)

    plot_equity_curve(result.equity_curve, osp.join(save_folder, "equity_curve.png"))
    plot_all_symbols(result.symbol_execution, result.trades, osp.join(save_folder, "trades_on_candles"))

    m = result.metrics
    lines = [
        "# Backtest Summary",
        "",
        f"- Annual Return: {m.annual_return * 100:.2f}%",
        f"- Sharpe: {m.sharpe:.3f}",
        f"- Sortino: {m.sortino:.3f}",
        f"- Calmar: {m.calmar:.3f}",
        f"- Max Drawdown: {m.max_drawdown * 100:.2f}%",
        f"- Win Rate: {m.win_rate * 100:.2f}%",
        f"- Profit Factor: {m.profit_factor:.3f}",
        f"- Avg Trade %: {m.avg_trade_pct:.3f}%",
        f"- Exposure: {m.exposure * 100:.2f}%",
        f"- Turnover: {m.turnover:.3f}x",
        f"- Trades: {m.trade_count}",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_optimizer_report(opt_result: dict, save_folder: str):
    """将优化结果写出为参数文件、trial 明细和汇总文件。"""
    os.makedirs(save_folder, exist_ok=True)

    if "all_trials" in opt_result:
        opt_result["all_trials"].to_csv(osp.join(save_folder, "all_trials.csv"), index=False)
    if "walk_forward_summary" in opt_result:
        opt_result["walk_forward_summary"].to_csv(osp.join(save_folder, "walk_forward_summary.csv"), index=False)

    payload = {k: v for k, v in opt_result.items() if k not in {"all_trials", "walk_forward_summary"}}
    with open(osp.join(save_folder, "optimizer_result.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if "best_params" in payload:
        with open(osp.join(save_folder, "best_params.yml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(payload["best_params"], f, allow_unicode=True, sort_keys=False)
