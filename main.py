import sys

from nofx.trader import AutoTrader


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("错误: 缺少第一个位置参数! 用法: python main.py <config-path>")

    config_path = sys.argv[1]
    trader = AutoTrader(config_path=config_path)
    trader.run()
