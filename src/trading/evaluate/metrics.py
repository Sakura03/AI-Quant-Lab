from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trading.data.catalog import timeframe_to_minutes
from trading.domain.types import Metrics


def max_drawdown_from_equity(equity: pd.Series) -> float:
    """Compute max drawdown from equity curve."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(abs(dd.min())) if len(dd) else 0.0


def compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, execution_timeframe: str) -> Metrics:
    """Compute return/risk/trade metrics from equity and trade history."""
    if equity_df.empty:
        return Metrics()

    eq = equity_df["equity"].astype(float)
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    period_min = timeframe_to_minutes(execution_timeframe)
    periods_per_year = int((365 * 24 * 60) / period_min)

    ann_return = 0.0
    sharpe = 0.0
    sortino = 0.0
    calmar = 0.0

    if len(eq) > 1 and eq.iloc[0] > 0:
        total_return = eq.iloc[-1] / eq.iloc[0]
        years = max((len(eq) - 1) / periods_per_year, 1e-9)
        ann_return = float(total_return ** (1.0 / years) - 1.0)

    if len(rets) > 5:
        mean_ret = float(rets.mean())
        std_ret = float(rets.std(ddof=0))
        if std_ret > 0:
            sharpe = float(math.sqrt(periods_per_year) * mean_ret / std_ret)

        # Downside deviation for Sortino should include non-negative returns as zero.
        downside = rets.clip(upper=0.0)
        downside_std = float(downside.std(ddof=0)) if len(downside) else 0.0
        if downside_std > 0:
            sortino = float(math.sqrt(periods_per_year) * mean_ret / downside_std)

    mdd = max_drawdown_from_equity(eq)
    if mdd > 0:
        calmar = ann_return / mdd

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

    exposure = float((equity_df["open_positions"] > 0).mean())
    mean_equity = float(eq.mean()) if len(eq) else 0.0
    turnover = float(equity_df["turnover"].iloc[-1] / mean_equity) if mean_equity > 0 else 0.0

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


def score_metrics(
    train_metrics: Metrics,
    eval_metrics: Metrics,
    max_symbol_share: float,
    objective_weights: dict[str, float],
    hard_limits: dict[str, float],
    min_trades: int,
) -> float:
    """Score one evaluation result under objective weights and hard constraints."""
    mdd_limit = float(hard_limits.get("max_drawdown", 0.15))
    if eval_metrics.max_drawdown > mdd_limit:
        return -1e9

    overfit_gap = abs(train_metrics.sharpe - eval_metrics.sharpe)
    low_trade_penalty = 0.0 if eval_metrics.trade_count >= min_trades else (min_trades - eval_metrics.trade_count) * 0.05

    return (
        objective_weights.get("sharpe", 1.0) * eval_metrics.sharpe
        - objective_weights.get("max_drawdown", 2.0) * eval_metrics.max_drawdown
        - objective_weights.get("turnover", 0.1) * eval_metrics.turnover
        - objective_weights.get("overfit_gap", 0.2) * overfit_gap
        - objective_weights.get("concentration", 0.5) * max(0.0, max_symbol_share - 0.35)
        - low_trade_penalty
    )
