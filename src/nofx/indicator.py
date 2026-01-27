from typing import Tuple, Dict, Any

import pandas as pd
import talib

from .structs import MarketData


def heikinashi(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = df.copy()
    df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    # ha open
    df.at[0, "ha_open"] = (df.at[0, "open"] + df.at[0, "close"]) / 2
    for i in range(1, len(df)):
        df.at[i, "ha_open"] = (df.at[i - 1, "ha_open"] + df.at[i - 1, "ha_close"]) / 2

    df["ha_high"] = df.loc[:, ["high", "ha_open", "ha_close"]].max(axis=1)
    df["ha_low"] = df.loc[:, ["low", "ha_open", "ha_close"]].min(axis=1)

    return df["ha_open"], df["ha_high"], df["ha_low"], df["ha_close"]


def ichimoku(
        df: pd.DataFrame,
        conversion_line_period: int = 9,
        base_line_period: int = 26,
        lagging_span: int = 52,
        displacement: int = 26,
        use_heikinashi: bool = False,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if use_heikinashi:
        _, high, low, _ = heikinashi(df)
    else:
        _, high, low, _ = df["open"], df["high"], df["low"], df["close"]

    # 转换线 (Tenkan-sen)
    tenkan_sen = (
        high.rolling(window=conversion_line_period).max()
        + low.rolling(window=conversion_line_period).min()
    ) / 2

    # 基准线 (Kijun-sen)
    kijun_sen = (
        high.rolling(window=base_line_period).max()
        + low.rolling(window=base_line_period).min()
    ) / 2

    leading_senkou_span_a = (tenkan_sen + kijun_sen) / 2

    leading_senkou_span_b = (
        high.rolling(window=lagging_span).max()
        + low.rolling(window=lagging_span).min()
    ) / 2

    # 先行A线 (Senkou Span A)
    senkou_span_a = leading_senkou_span_a.shift(displacement - 1)

    # 先行B线 (Senkou Span B)
    senkou_span_b = leading_senkou_span_b.shift(displacement - 1)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b


def add_indicator(df: pd.DataFrame, indicator_param: Dict[str, Any]):
    name, params, col = indicator_param["name"], indicator_param["params"], indicator_param["display_name"]

    if name == "SMA":
        df[col] = talib.SMA(df["close"], **params)

    elif name == "EMA":
        df[col] = talib.EMA(df["close"], **params)

    elif name == "STDDEV":
        df[col] = talib.STDDEV(df["close"], **params)

    elif name == "BBANDS":
        upper, mid, lower = col
        df[upper], df[mid], df[lower] = talib.BBANDS(df["close"], **params)

    elif name == "MACD":
        dif, dea, macd = col
        df[dif], df[dea], df[macd] = talib.MACD(df["close"], **params)

    elif name == "RSI":
        df[col] = talib.RSI(df["close"], **params)

    elif name == "ATR":
        df[col] = talib.ATR(df["high"], df["low"], df["close"], **params)

    elif name == "ADX":
        df[col] = talib.ADX(df["high"], df["low"], df["close"], **params)

    elif name == "KDJ":
        k, d, j = col
        df[k], df[d] = talib.STOCH(df["high"], df["low"], df["close"], **params)
        df[j] = 3 * df[k] - 2 * df[d]

    elif name == "HEIKINASHI":
        ha_open, ha_high, ha_low, ha_close = col
        df[ha_open], df[ha_high], df[ha_low], df[ha_close] = heikinashi(df)

    elif name == "ICHIMOKU":
        tenkan, kijun, senkou_a, senkou_b = col
        df[tenkan], df[kijun], df[senkou_a], df[senkou_b] = ichimoku(df, **params)

    elif name == "VOLSMA":
        df[col] = talib.SMA(df["volume"], **params)

    elif name == "VOLEMA":
        df[col] = talib.EMA(df["volume"], **params)

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
