from __future__ import annotations

import pandas as pd

from trading.evaluate.metrics import compute_metrics


def infer_regime_labels(equity_df: pd.DataFrame) -> pd.Series:
    """Label each equity row into bull/bear/sideways using rolling 1-day return."""
    if equity_df.empty:
        return pd.Series(dtype=object)
    out = equity_df[["timestamp", "equity"]].copy()
    out["ret_1d"] = out["equity"].pct_change(1440).fillna(0.0)

    labels = []
    for x in out["ret_1d"].to_list():
        if x >= 0.01:
            labels.append("bull")
        elif x <= -0.01:
            labels.append("bear")
        else:
            labels.append("sideways")
    return pd.Series(labels, index=equity_df.index)


def regime_breakdown(equity_df: pd.DataFrame, trades_df: pd.DataFrame, exec_tf: str) -> pd.DataFrame:
    """Compute per-regime performance metrics over stitched OOS equity."""
    if equity_df.empty:
        return pd.DataFrame(columns=["regime", "sharpe", "max_drawdown", "annual_return", "trade_count"])

    labels = infer_regime_labels(equity_df)
    rows = []
    for regime in ["bull", "bear", "sideways"]:
        idx = labels[labels == regime].index
        if len(idx) < 30:
            continue
        sub_eq = equity_df.loc[idx].copy().reset_index(drop=True)
        # Trade slicing by time overlap.
        t0 = sub_eq["timestamp"].min()
        t1 = sub_eq["timestamp"].max()
        if trades_df.empty:
            sub_trades = trades_df.copy()
        else:
            exit_ts = pd.to_datetime(trades_df["exit_time"]) if "exit_time" in trades_df.columns else pd.Series(dtype="datetime64[ns]")
            sub_trades = trades_df[(exit_ts >= t0) & (exit_ts <= t1)].copy() if len(exit_ts) else trades_df.iloc[0:0].copy()
        m = compute_metrics(sub_eq, sub_trades, exec_tf)
        rows.append(
            {
                "regime": regime,
                "annual_return": m.annual_return,
                "sharpe": m.sharpe,
                "max_drawdown": m.max_drawdown,
                "trade_count": m.trade_count,
            }
        )

    return pd.DataFrame(rows)
