from functools import wraps
from typing import Tuple, List, Optional, Type, Any

import time
import os.path as osp
import pandas as pd

from .enums import PositionSide


def timeframe_to_seconds(timeframe: str) -> int:
    """
        将 ccxt/自定义的时间周期字符串转换为秒数
        支持: s, m, min, h, d, w
        示例: "5m" -> 300, "1h" -> 3600
    """
    if timeframe.endswith("s"):
        amount = int(timeframe[:-1])
        return amount
    elif timeframe.endswith("m"):
        amount = int(timeframe[:-1])
        return amount * 60
    elif timeframe.endswith("min"):
        amount = int(timeframe[:-3])
        return amount * 60
    elif timeframe.endswith("h"):
        amount = int(timeframe[:-1])
        return amount * 3600
    elif timeframe.endswith("d"):
        amount = int(timeframe[:-1])
        return amount * 86400
    elif timeframe.endswith("w"):
        amount = int(timeframe[:-1])
        return amount * 604800
    else:
        raise ValueError(f"无效的时间周期: {timeframe:s}")


def format_timeframe(timeframe: str) -> str:
    """
        将 ccxt/自定义的时间周期字符串转换为中文时间
        支持: s, m, min, h, d, w
        示例: "5m" -> "5分钟", "1h" -> "1小时"
    """
    if timeframe.endswith("s"):
        return timeframe.replace("s", "秒")
    elif timeframe.endswith("m"):
        return timeframe.replace("m", "分钟")
    elif timeframe.endswith("min"):
        return timeframe.replace("min", "分钟")
    elif timeframe.endswith("h"):
        return timeframe.replace("h", "小时")
    elif timeframe.endswith("d"):
        return timeframe.replace("d", "天")
    elif timeframe.endswith("w"):
        return timeframe.replace("w", "周")
    else:
        raise ValueError(f"无效的时间周期: {timeframe:s}")


def format_time_interval(seconds: int, return_second: bool = False) -> str:
    """
        将秒数格式化为中文可读时间
        示例: 3700 -> "1小时1分钟40秒"
    """
    parts = []
    minutes = seconds // 60
    seconds = seconds % 60
    parts.append(f"{seconds:d}秒")

    hours = 0
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
    parts.append(f"{minutes:d}分钟")

    days = 0
    if hours >= 24:
        days = hours // 24
        hours = hours % 24

    if hours > 0:
        parts.append(f"{hours:d}小时")

    if days > 0:
        parts.append(f"{days:d}天")

    if not return_second:
        parts = parts[1:]

    return "".join(reversed(parts))


def format_symbol(symbol: str) -> str:
    """
        将 symbol 格式化为标准形式: XXX/USDT
        可接受输入格式:
            - "BTC"
            - "BTCUSDT"
            - "BTC/USDT"
            - "BTC/USDT:USDT"
            - "BTC-USDT"
            - "BTCUSDT:USDT"

        输出统一为:
            "BTC/USDT"
    """
    if not isinstance(symbol, str):
        raise TypeError("symbol的类型不是string")

    s = symbol.upper().strip()

    # 1. 去掉多余后缀，如 ":USDT"
    if ":" in s:
        s = s.split(":")[0]

    # 2. 替换常见分隔符为统一的 '/'
    s = s.replace("-", "/")

    # 3. 如果已经是 XXX/USDT 格式
    if "/" in s:
        base = s.split("/")[0]

    # 4. 如果是连续字符串，如 "BTCUSDT"
    elif s.endswith("USDT"):
        base = s[:-4]

    # 5. 输入只有币种，如 "BTC"
    else:
        base = s

    return f"{base:s}/USDT"


def parse_dataframe(data_path: str) -> Optional[pd.DataFrame]:
    if not osp.isfile(data_path):
        return None

    df = pd.read_feather(data_path)
    df["date"] = df["date"].dt.tz_localize(None)
    df = df.rename(columns={"date": "timestamp"})
    return df


def infer_timeframe(df: pd.DataFrame) -> pd.Timedelta:
    """
        根据 timestamp 列推断 DataFrame 的时间周期（如 1m, 5m, 4h 等）
        返回一个 pandas.Timedelta
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame须包含'timestamp'列")

    ts = df["timestamp"].sort_values().drop_duplicates()

    if len(ts) < 2:
        raise ValueError("DataFrame至多只有一行数据")

    # 计算相邻时间差
    diffs = ts.diff().dropna()

    # 取最常出现的 diff 作为时间周期
    return diffs.mode().iloc[0]


def truncate_dataframe(df: pd.DataFrame, time: pd.Timestamp, limit: Optional[int] = None) -> pd.DataFrame:
    """
        截取 DataFrame 中 timestamp <= time 的部分，并取最后 limit 行

        参数:
            df: 包含 "timestamp" 列的 pandas DataFrame
            time: pd.Timestamp, 用于筛选时间
            limit: int, 限制返回的行数, 若为 None, 则不限制

        返回:
            pd.DataFrame: 截取后的新 DataFrame
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame须包含'timestamp'列")

    timeframe = infer_timeframe(df)

    # 过滤出 timestamp <= time 的行
    filtered = df[df["timestamp"] <= time - timeframe]

    return filtered.tail(limit) if limit else filtered


def fetch_lastest_data(df: pd.DataFrame, time: pd.Timestamp, cols: List[str]) -> List[Any]:
    """
        从 DataFrame 中获取指定时间点 (time) 之前最新一条可用的数据

        参数：
            df : pd.DataFrame
                必须包含 'timestamp' 和 'close' 列
            time : pd.Timestamp
                要查找的目标时间
            cols : List[str]
                所要数据的列名

        返回：
            列表, 列表元素类型:
                Any  : time 之前最新的某列数据
                None : 如果 DataFrame 所有行的 timestamp 都大于 time
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame须包含'timestamp'列")

    filtered = truncate_dataframe(df, time)

    if len(filtered) == 0:
        return [None for _ in cols]

    # 取最后一行的 close 值
    return filtered.iloc[-1][cols].values.tolist()


def calculate_liquidation_price(side: PositionSide, entry_price: float, leverage: int) -> float:
    """
        计算 USDT-M 合约的爆仓价（不考虑手续费）
    """

    if leverage <= 0:
        raise ValueError("杠杆倍数必须大于0")

    if side == PositionSide.Long:
        # 多单爆仓价
        return entry_price * (1 - 1 / leverage)

    elif side == PositionSide.Short:
        # 空单爆仓价
        return entry_price * (1 + 1 / leverage)

    else:
        raise ValueError(f"无效的持仓类型: {side.name:s}")


def retry(
        max_retries: int,
        delay: float,
        exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        output: Any = None,
        logger_attr: str = "warning",
        raise_if_fail: bool = False,
):
    """
        通用 retry 装饰器：
        - max_retries: 最大重试次数
        - delay: 延迟
        - exceptions: 捕获的异常
        - output: 当重试次数用完时的返回值
        - logger_attr
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]  # self是BaseClassWithLogger的派生类
            log = getattr(self, logger_attr, None)
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    is_last = i == max_retries - 1
                    if log:
                        msg = f"[重试: {i+1:d}/{max_retries:d}] {func.__name__:s} 失败: {e}"
                        if not is_last:
                            msg += f", {delay:.2f}秒后重试"
                        log(msg)
                    if is_last:
                        if raise_if_fail:
                            raise ConnectionError(f"{func.__name__:s}: 重试{max_retries:d}次均失败")
                        return output
                    time.sleep(delay)
        return wrapper
    return decorator
