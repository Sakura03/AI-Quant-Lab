from dataclasses import asdict

import argparse
import os
import os.path as osp
import pandas as pd

from nofx.config import Config
from nofx.logger import setup_logger
from nofx.exchange import Exchange
from nofx.utils import timeframe_to_seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载回测数据的脚本")
    parser.add_argument("config", type=str, help="配置文件路径")
    parser.add_argument("--start-time", default=None, type=str, help="数据的起始时间")
    parser.add_argument("--end-time", default=None, type=str, help="数据的结束时间")
    parser.add_argument("--symbols", nargs="*", default=[], help="交易对列表")
    parser.add_argument("--timeframes", nargs="*", default=[], help="时间周期列表")
    args = parser.parse_args()

    config = Config.parse_config_file(args.config)
    logger = setup_logger("generate_data")

    os.makedirs(config.backtest.data_folder, exist_ok=True)

    symbols = args.symbols or config.trader.symbols
    timeframes = args.timeframes or list(config.indicators.keys())
    if "1m" not in timeframes:
        timeframes.append("1m")

    start_time = args.start_time or config.backtest.start_time
    end_time = args.end_time or config.backtest.end_time
    start_time = pd.to_datetime(start_time, format="%Y%m%d-%H%M%S")
    end_time = pd.to_datetime(end_time, format="%Y%m%d-%H%M%S")

    exchange = Exchange(**asdict(config.exchange), logger=logger)

    for symbol in symbols:
        symbol_future = symbol.replace("/", "")
        # Funding Rate
        df = exchange.fetch_history_funding_rate(symbol_future, start_time=start_time, end_time=end_time)
        df["timestamp"] = df["timestamp"].astype("int64") // int(1e6)
        filename = symbol.replace("/", "_") + "-funding-rate.feather"
        save_path = osp.join(config.backtest.data_folder, filename)
        df.to_feather(save_path)

        # OHLCV
        for timeframe in timeframes:
            # 多存1000个周期的数据, 为了计算技术指标
            period = pd.Timedelta(seconds=timeframe_to_seconds(timeframe))
            df = exchange.fetch_history_ohlcv_df(symbol_future, timeframe, start_time=start_time - 1000 * period, end_time=end_time)
            df["timestamp"] = df["timestamp"].astype("int64") // int(1e6)
            filename = symbol.replace("/", "_") + "-" + timeframe + ".feather"
            save_path = osp.join(config.backtest.data_folder, filename)
            df.to_feather(save_path)
