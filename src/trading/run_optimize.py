from __future__ import annotations

import argparse

from .config import TradingConfig
from .optimizer import WalkForwardOptimizer
from .report import build_run_folder, write_optimizer_report


def parse_args() -> argparse.Namespace:
    """解析优化命令行参数。"""
    p = argparse.ArgumentParser(description="Run walk-forward strategy optimization")
    p.add_argument("--config", required=True, help="Path to trading config YAML")
    return p.parse_args()


def main():
    """优化 CLI 入口：执行 walk-forward 搜索并输出最优结果。"""
    args = parse_args()
    config = TradingConfig.parse_config_file(args.config)

    optimizer = WalkForwardOptimizer(config)
    result = optimizer.run()

    save_folder = build_run_folder(config.output.save_folder, "optimize")
    write_optimizer_report(result, save_folder)

    metrics = result["best_test_metrics"]
    print(f"Saved to: {save_folder}")
    print(
        f"Best Window={result['best_window_id']} | Score={result['best_test_score']:.4f} | "
        f"Sharpe={metrics['sharpe']:.3f} | MaxDD={metrics['max_drawdown']*100:.2f}%"
    )


if __name__ == "__main__":
    main()
