from __future__ import annotations

import argparse

from .backtest import Backtester
from .config import TradingConfig
from .report import build_run_folder, write_backtest_report


def parse_args() -> argparse.Namespace:
    """解析回测命令行参数。"""
    p = argparse.ArgumentParser(description="Run technical-analysis backtest")
    p.add_argument("--config", required=True, help="Path to trading config YAML")
    return p.parse_args()


def main():
    """回测 CLI 入口：读取配置、运行回测、写出结果摘要。"""
    args = parse_args()
    config = TradingConfig.parse_config_file(args.config)

    backtester = Backtester(config)
    result = backtester.run()

    save_folder = build_run_folder(config.output.save_folder, "backtest")
    write_backtest_report(result, config, save_folder)

    m = result.metrics
    print(f"Saved to: {save_folder}")
    print(
        f"Sharpe={m.sharpe:.3f} | MaxDD={m.max_drawdown*100:.2f}% | "
        f"AnnRet={m.annual_return*100:.2f}% | Trades={m.trade_count}"
    )


if __name__ == "__main__":
    main()
