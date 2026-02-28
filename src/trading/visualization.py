from __future__ import annotations

from typing import Iterable

import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def plot_equity_curve(equity_df: pd.DataFrame, save_path: str):
    """绘制资金曲线与回撤面积图并保存为图片。"""
    if equity_df.empty:
        return

    df = equity_df.copy()
    df = df.sort_values("timestamp")
    df["drawdown"] = df["equity"] / df["equity"].cummax() - 1.0

    fig = plt.figure(figsize=(18, 9))
    ax1 = plt.gca()
    ax1.plot(df["timestamp"], df["equity"], label="equity", linewidth=2)
    ax1.plot(df["timestamp"], df["cash"], label="cash", linewidth=1.5, alpha=0.8)
    ax1.set_ylabel("USDT")
    ax1.set_xlabel("Time")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.fill_between(df["timestamp"], df["drawdown"] * 100.0, 0.0, where=(df["drawdown"] < 0), alpha=0.25, color="tab:red")
    ax2.set_ylim(-100, 0)
    ax2.set_ylabel("Drawdown (%)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_trades_on_candles(
    symbol: str,
    execution_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    save_path: str,
    max_bars: int = 5000,
):
    """绘制单个交易对的 K 线及交易进出场标记并保存为 HTML。"""
    if execution_df.empty:
        return

    df = execution_df.sort_values("timestamp").tail(max_bars).copy()
    trades = trades_df[trades_df["symbol"] == symbol].copy() if not trades_df.empty else pd.DataFrame()

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=symbol,
            increasing_line_color="#d62728",
            decreasing_line_color="#2ca02c",
        )
    )

    if not trades.empty:
        entries = trades[["entry_time", "entry_price", "side"]].copy()
        exits = trades[["exit_time", "exit_price", "pnl_pct", "exit_reason"]].copy()

        fig.add_trace(
            go.Scatter(
                x=entries["entry_time"],
                y=entries["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-up", color="#ff7f0e", size=10),
                name="entry",
                text=entries["side"],
            )
        )

        fig.add_trace(
            go.Scatter(
                x=exits["exit_time"],
                y=exits["exit_price"],
                mode="markers",
                marker=dict(symbol="triangle-down", color="#1f77b4", size=10),
                name="exit",
                text=exits.apply(lambda r: f"{r['pnl_pct']:+.2f}% | {r['exit_reason']}", axis=1),
            )
        )

    fig.update_layout(
        title=f"{symbol} Trades",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)


def plot_all_symbols(symbol_execution: dict[str, pd.DataFrame], trades_df: pd.DataFrame, save_folder: str):
    """为所有交易对批量输出交易标记 K 线图。"""
    for symbol, df in symbol_execution.items():
        filename = symbol.replace("/", "_") + ".html"
        plot_trades_on_candles(symbol, df, trades_df, osp.join(save_folder, filename))
