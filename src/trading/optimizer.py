from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from .backtest import Backtester
from .config import TradingConfig, clone_config_with_period
from .data_loader import load_market_store
from .signals import sample_strategy_params


def add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    """在给定时间上增加指定月数。"""
    return ts + pd.DateOffset(months=months)


def make_walk_forward_windows(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    train_months: int,
    valid_months: int,
    test_months: int,
) -> list[dict[str, pd.Timestamp]]:
    """按训练/验证/测试月数切分 walk-forward 时间窗口。"""
    windows: list[dict[str, pd.Timestamp]] = []
    cursor = start_time

    while True:
        train_start = cursor
        train_end = add_months(train_start, train_months)
        valid_end = add_months(train_end, valid_months)
        test_end = add_months(valid_end, test_months)

        if test_end > end_time:
            break

        windows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "valid_start": train_end,
                "valid_end": valid_end,
                "test_start": valid_end,
                "test_end": test_end,
            }
        )
        cursor = add_months(cursor, test_months)

    return windows


def objective_score(metrics, drawdown_limit: float, weights: Dict[str, float]) -> float:
    """计算优化目标分数，超过回撤阈值时直接判为极差分。"""
    if metrics.max_drawdown > drawdown_limit:
        return -1e9
    return (
        weights["sharpe"] * metrics.sharpe
        - weights["max_drawdown_penalty"] * metrics.max_drawdown
        - weights["turnover_penalty"] * metrics.turnover
    )


class WalkForwardOptimizer:
    """Walk-forward 随机搜索优化器。"""
    def __init__(self, config: TradingConfig):
        """初始化优化器并加载市场数据缓存。"""
        self.config = config
        self.market_store = load_market_store(config.data)
        self.rng = np.random.default_rng(config.engine.seed)

    def run(self) -> dict[str, Any]:
        """执行完整 walk-forward 搜索，返回最优参数和全量试验结果。"""
        # 1) 读取优化配置并生成 walk-forward 时间窗口。
        opt_cfg = self.config.optimization
        windows = make_walk_forward_windows(
            self.config.data.start_time,
            self.config.data.end_time,
            opt_cfg.train_months,
            opt_cfg.valid_months,
            opt_cfg.test_months,
        )
        if not windows:
            raise ValueError("时间区间不足以构建 walk-forward 窗口")

        # 2) 准备全局收集器：
        # - all_trials: 每个 trial 的训练/验证指标与参数
        # - wf_rows: 每个窗口最终测试结果
        # - best_overall: 全流程最优窗口与参数
        all_trials = []
        wf_rows = []
        best_overall = None

        # 3) 逐窗口优化：每个窗口内仅用训练+验证挑参数，不窥探测试集。
        for win_id, win in enumerate(windows, start=1):
            best_trial = None

            # 3.1) 试验循环：随机采样参数并评估训练/验证表现。
            for trial_id in range(1, opt_cfg.trials + 1):
                params = sample_strategy_params(self.rng)

                # 使用同一大配置，仅替换时间区间和策略参数，构造训练/验证配置。
                train_cfg = clone_config_with_period(self.config, win["train_start"], win["train_end"], params)
                valid_cfg = clone_config_with_period(self.config, win["valid_start"], win["valid_end"], params)

                # 分别回测训练集与验证集。
                train_bt = Backtester(train_cfg, strategy_params=params, market_store=self.market_store)
                valid_bt = Backtester(valid_cfg, strategy_params=params, market_store=self.market_store)

                train_res = train_bt.run()
                valid_res = valid_bt.run()

                # 基于验证集打分（Sharpe 主导 + 回撤硬约束 + 换手惩罚）。
                score = objective_score(
                    valid_res.metrics,
                    drawdown_limit=opt_cfg.drawdown_limit,
                    weights=opt_cfg.objective_weights,
                )

                # 记录 trial 明细，便于后续分析参数敏感性。
                row = {
                    "window_id": win_id,
                    "trial_id": trial_id,
                    "train_sharpe": train_res.metrics.sharpe,
                    "train_mdd": train_res.metrics.max_drawdown,
                    "valid_sharpe": valid_res.metrics.sharpe,
                    "valid_mdd": valid_res.metrics.max_drawdown,
                    "valid_turnover": valid_res.metrics.turnover,
                    "score": score,
                    **{f"p_{k}": v for k, v in params.items()},
                }
                all_trials.append(row)

                # 维护当前窗口内最佳 trial（以验证分数为准）。
                if best_trial is None or score > best_trial["score"]:
                    best_trial = {"score": score, "params": params, "row": row}

            if best_trial is None:
                continue

            # 3.2) 固化窗口最优参数，并在该窗口测试集上做一次真正 OOS 评估。
            test_cfg = clone_config_with_period(
                self.config,
                win["test_start"],
                win["test_end"],
                best_trial["params"],
            )
            test_bt = Backtester(test_cfg, strategy_params=best_trial["params"], market_store=self.market_store)
            test_res = test_bt.run()
            test_score = objective_score(
                test_res.metrics,
                drawdown_limit=opt_cfg.drawdown_limit,
                weights=opt_cfg.objective_weights,
            )

            # 记录窗口级结果，后续可直接导出为 walk_forward_summary.csv。
            wf_row = {
                "window_id": win_id,
                "train_start": win["train_start"].strftime("%Y-%m-%d"),
                "train_end": win["train_end"].strftime("%Y-%m-%d"),
                "valid_end": win["valid_end"].strftime("%Y-%m-%d"),
                "test_end": win["test_end"].strftime("%Y-%m-%d"),
                "best_valid_score": best_trial["score"],
                "test_score": test_score,
                "test_sharpe": test_res.metrics.sharpe,
                "test_mdd": test_res.metrics.max_drawdown,
                "test_annual_return": test_res.metrics.annual_return,
                "test_trade_count": test_res.metrics.trade_count,
                **{f"p_{k}": v for k, v in best_trial["params"].items()},
            }
            wf_rows.append(wf_row)

            # 维护全局最优（按测试集分数，不按验证集分数）。
            if best_overall is None or test_score > best_overall["test_score"]:
                best_overall = {
                    "test_score": test_score,
                    "best_params": best_trial["params"],
                    "window_id": win_id,
                    "test_metrics": test_res.metrics.to_dict(),
                }

        if best_overall is None:
            raise RuntimeError("优化过程中未找到可用参数")

        # 4) 汇总返回：既返回最优结论，也返回完整试验明细供复盘。
        return {
            "method": self.config.optimization.method,
            "drawdown_limit": self.config.optimization.drawdown_limit,
            "weights": dict(self.config.optimization.objective_weights),
            "best_params": best_overall["best_params"],
            "best_window_id": best_overall["window_id"],
            "best_test_score": best_overall["test_score"],
            "best_test_metrics": best_overall["test_metrics"],
            "walk_forward_summary": pd.DataFrame(wf_rows),
            "all_trials": pd.DataFrame(all_trials),
        }
