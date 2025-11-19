from typing import Tuple, Dict, Any

import pandas as pd
import talib

from .structs import MarketData


def ichimoku(high: pd.Series, low: pd.Series, conversion_line_period: int = 9, base_line_period: int = 26, lagging_span: int = 52, displacement: int = 26) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    # 转换线 (Tenkan-sen)
    tenkan_sen = (high.rolling(window=conversion_line_period).max() + low.rolling(window=conversion_line_period).min()) / 2

    # 基准线 (Kijun-sen)
    kijun_sen = (high.rolling(window=base_line_period).max() + low.rolling(window=base_line_period).min()) / 2

    # 先行A线 (Senkou Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

    # 先行B线 (Senkou Span B)
    senkou_span_b = ((high.rolling(window=lagging_span).max() + low.rolling(window=lagging_span).min()) / 2).shift(displacement)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b


def add_indicator(df: pd.DataFrame, indicator_param: Dict[str, Any]):
    name, params, col = indicator_param["name"], indicator_param["params"], indicator_param["display_name"]

    if name == "SMA":
        df[col] = talib.SMA(df["close"], **params)

    elif name == "EMA":
        df[col] = talib.EMA(df["close"], **params)

    elif name == "BBANDS":
        if len(col) == 2:
            upper, lower = col
            df[upper], _, df[lower] = talib.BBANDS(df["close"], **params)
        else:
            upper, mid, lower = col
            df[upper], df[mid], df[lower] = talib.BBANDS(df["close"], **params)

    elif name == "MACD":
        if isinstance(col, str):
            _, _, df[col] = talib.MACD(df["close"], **params)
        else:
            dif, dea, macd = col
            df[dif], df[dea], df[macd] = talib.MACD(df["close"], **params)

    elif name == "RSI":
        df[col] = talib.RSI(df["close"], **params)

    elif name == "ATR":
        df[col] = talib.ATR(df["high"], df["low"], df["close"], **params)

    elif name == "KDJ":
        k, d, j = col
        df[k], df[d] = talib.STOCH(df["high"], df["low"], df["close"], **params)
        df[j] = 3 * df[k] - 2 * df[d]

    elif name == "ICHIMOKU":
        tenkan, kijun, senkou_a, senkou_b = col
        df[tenkan], df[kijun], df[senkou_a], df[senkou_b] = ichimoku(df["high"], df["low"], **params)

    else:
        raise NotImplementedError(f"未知的技术指标: {name:s}")


def add_indicators(market_data: MarketData, indicators: Dict[str, Any]):
    for symbol, symbol_data in market_data.items():
        for timeframe, params in indicators.items():
            if timeframe not in symbol_data.data:
                raise KeyError(f"{symbol:s}缺失时间周期{timeframe:s}的数据")

            df = symbol_data.data[timeframe]
            for param in params:
                add_indicator(df, param)
