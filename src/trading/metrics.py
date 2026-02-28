from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .types import Metrics


def timeframe_to_minutes(timeframe: str) -> int:
    """将 `1m/1h/1d` 等周期字符串转换为分钟数。"""
    m = re.fullmatch(r"(\d+)([mhd])", timeframe.strip().lower())
    if not m:
        raise ValueError(f"无效时间周期: {timeframe}")
    value = int(m.group(1))
    unit = m.group(2)
    scale = {"m": 1, "h": 60, "d": 1440}[unit]
    return value * scale


def max_drawdown_from_equity(equity: pd.Series) -> float:
    """根据权益序列计算最大回撤（返回正值前需取绝对值）。"""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min()) if len(dd) > 0 else 0.0


def compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, execution_timeframe: str) -> Metrics:
    """从权益曲线与交易列表计算核心绩效指标。"""
    # 1) 空输入保护：无权益曲线时直接返回默认指标，避免后续索引报错。
    if equity_df.empty:
        return Metrics()

    # 2) 预处理收益序列：权益转 float，并计算逐周期收益率。
    eq = equity_df["equity"].astype(float)
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    # 3) 年化尺度：根据执行周期换算每年周期数，供年化类指标使用。
    period_min = timeframe_to_minutes(execution_timeframe)
    periods_per_year = int((365 * 24 * 60) / period_min)

    # 4) 指标默认值：在样本不足时保留 0，避免输出 NaN/inf。
    ann_return = 0.0
    sharpe = 0.0
    sortino = 0.0
    calmar = 0.0

    # 5) 年化收益：按起止权益和覆盖年数计算 CAGR。
    if len(eq) > 1 and eq.iloc[0] > 0:
        total_return = eq.iloc[-1] / eq.iloc[0]
        years = max((len(eq) - 1) / periods_per_year, 1e-9)
        ann_return = float(total_return ** (1.0 / years) - 1.0)

    # 6) 风险调整收益：先算 Sharpe，再以负收益波动算 Sortino。
    if len(rets) > 5:
        mean_ret = rets.mean()
        std_ret = rets.std(ddof=0)
        if std_ret > 0:
            sharpe = float(math.sqrt(periods_per_year) * mean_ret / std_ret)

        downside = rets[rets < 0]
        downside_std = downside.std(ddof=0)
        if downside_std > 0:
            sortino = float(math.sqrt(periods_per_year) * mean_ret / downside_std)

    # 7) 回撤相关：最大回撤用于约束风险，并计算 Calmar 比率。
    mdd = abs(max_drawdown_from_equity(eq))
    if mdd > 0:
        calmar = ann_return / mdd

    # 8) 交易级统计：胜率、盈亏比、平均单笔收益率。
    trade_count = int(len(trades_df))
    win_rate = 0.0
    profit_factor = 0.0
    avg_trade_pct = 0.0
    if trade_count > 0:
        pnl = trades_df["pnl_usd"].astype(float)
        wins = pnl[pnl > 0].sum()
        losses = -pnl[pnl < 0].sum()
        win_rate = float((pnl > 0).mean())
        profit_factor = float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0
        avg_trade_pct = float(trades_df["pnl_pct"].astype(float).mean())

    # 9) 资金利用率统计：暴露率与换手率（总成交额 / 平均权益）。
    exposure = float((equity_df["open_positions"] > 0).mean())
    mean_equity = float(eq.mean()) if len(eq) else 0.0
    turnover = float(equity_df["turnover"].iloc[-1] / mean_equity) if mean_equity > 0 else 0.0

    # 10) 汇总返回：统一通过 Metrics 数据结构输出。
    return Metrics(
        annual_return=ann_return,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=mdd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_pct=avg_trade_pct,
        exposure=exposure,
        turnover=turnover,
        trade_count=trade_count,
    )
