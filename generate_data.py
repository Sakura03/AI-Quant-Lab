from dataclasses import asdict

import os
import os.path as osp
import pandas as pd

from nofx.config import Config
from nofx.logger import setup_logger
from nofx.exchange import Exchange
from nofx.utils import timeframe_to_seconds


if __name__ == "__main__":
    config_path = "configs/default.yml"
    config = Config.parse_config_file(config_path)
    logger = setup_logger("generate_data")

    os.makedirs(config.backtest.data_folder, exist_ok=True)

    symbols = config.trader.symbols
    timeframes = list(config.indicators.keys())
    if "1m" not in timeframes:
        timeframes.append("1m")

    exchange = Exchange(**asdict(config.exchange), logger=logger)

    for symbol in config.trader.symbols:
        symbol_future = symbol.replace("/", "")
        # Funding Rate
        df = exchange.fetch_history_funding_rate(symbol_future, start_time=config.backtest.start_time, end_time=config.backtest.end_time)
        df["timestamp"] = df["timestamp"].astype("int64") // int(1e6)
        filename = symbol.replace("/", "_") + "-funding-rate.feather"
        save_path = osp.join(config.backtest.data_folder, filename)
        df.to_feather(save_path)

        # OHLCV
        for timeframe in timeframes:
            # 多存1000个周期的数据, 为了计算技术指标
            period = pd.Timedelta(seconds=timeframe_to_seconds(timeframe))
            start_time = config.backtest.start_time - 1000 * period
            df = exchange.fetch_history_ohlcv_df(symbol_future, timeframe, start_time=start_time, end_time=config.backtest.end_time)
            df["timestamp"] = df["timestamp"].astype("int64") // int(1e6)
            filename = symbol.replace("/", "_") + "-" + timeframe + ".feather"
            save_path = osp.join(config.backtest.data_folder, filename)
            df.to_feather(save_path)
