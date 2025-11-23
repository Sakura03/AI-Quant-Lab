from typing import List, Optional, Any

import os.path as osp
import pandas as pd


def parse_dataframe(data_path: str) -> Optional[pd.DataFrame]:
    if not osp.isfile(data_path):
        return None

    df = pd.read_feather(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def infer_period(df: pd.DataFrame) -> pd.Timedelta:
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


def truncate_dataframe(
        df: pd.DataFrame,
        end_time: pd.Timestamp,
        start_time: Optional[pd.Timestamp] = None,
        limit: Optional[int] = None,
        available_until_next_period: bool = True,
) -> pd.DataFrame:
    """
        截取 DataFrame 中 start_time <= timestamp <= end_time 的部分，并取最后 limit 行

        参数:
            df: 包含 "timestamp" 列的 pandas DataFrame
            end_time: pd.Timestamp, 结束时间
            start_time: pd.Timestamp, 开始时间, 若为None, 则不做筛选
            limit: int, 限制返回的行数, 若为 None, 则不限制
            available_until_next_period: bool
                如果为 True, 则表示 df 中的每一行数据直到下一周期的开始时刻才能得到 (如 OHLCV 数据);
                否则, 则表示 df 中的每一行数据在本周期的开始时刻就能得到 (如 funding rate数据).

        返回:
            pd.DataFrame: 截取后的新 DataFrame
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame须包含'timestamp'列")

    if available_until_next_period:
        period = infer_period(df)
        end_time = end_time - period
        if start_time:
            start_time = start_time - period

    # 过滤出 timestamp <= time 的行
    filtered = df[df["timestamp"] <= end_time]
    if start_time:
        filtered = filtered[filtered["timestamp"] > start_time]

    return filtered.tail(limit) if limit else filtered


def fetch_lastest_data(df: pd.DataFrame, time: pd.Timestamp, cols: List[str], available_until_next_period: bool = True) -> List[Any]:
    """
        从 DataFrame 中获取指定时间点 (time) 之前最新一条可用的数据

        参数：
            df : pd.DataFrame
                必须包含 'timestamp' 和 'close' 列
            time : pd.Timestamp
                要查找的目标时间
            cols : List[str]
                所要数据的列名
            available_until_next_period: bool
                见函数 truncate_dataframe 的说明

        返回：
            列表, 列表元素类型:
                Any  : time 之前最新的某列数据
                None : 如果 DataFrame 所有行的 timestamp 都大于 time
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame须包含'timestamp'列")

    filtered = truncate_dataframe(df, time, available_until_next_period=available_until_next_period)

    if len(filtered) == 0:
        return [None for _ in cols]

    # 取最后一行的 close 值
    return filtered.iloc[-1][cols].values.tolist()
