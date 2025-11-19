from typing import List, Dict

import os
import os.path as osp
import json
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .dataframe import truncate_dataframe


def visualize_snapshots(folder: str, symbols: List[str], save_path: str):
    df = pd.DataFrame(columns=["equity", "available"] + symbols)
    df.index.name = "timestamp"   # 设置 index 名称

    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue

        with open(osp.join(folder, filename), "r", encoding="utf-8") as f:
            snapshot = json.load(f)

        timestamp = pd.to_datetime(snapshot["time"], format="%Y-%m-%d %H:%M:%S")
        equity = snapshot["balance"]["total_wallet_balance"] + snapshot["balance"]["total_unrealized_profit"]
        available = snapshot["balance"]["available_balance"]

        positions = [0.0 for _ in symbols]
        for position in snapshot["positions"]:
            symbol = position["symbol"]
            assert symbol in symbols
            i = symbols.index(symbol)
            positions[i] = position["margin_used"] + position["unrealized_pnl"]

        df.loc[timestamp] = [equity, available] + positions

    visualize_funding_curve(df, save_path, funding_col="equity", extra_cols=["available"] + symbols)


def visualize_funding_curve(df: pd.DataFrame, save_path: str, funding_col: str, extra_cols: List[str] = []):
    df = df.sort_index(ascending=True)
    df["drawdown"] = df[funding_col] / df[funding_col].cummax() - 1.0

    fig = plt.figure(figsize=(25, 12))

    ax1 = plt.gca()
    ax1.plot(df.index, df[funding_col], label="funding", linewidth=2)
    for c in extra_cols:
        ax1.plot(df.index, df[c], label=c, linewidth=2)
    ax1.set_ylabel("Equity (USDT)")
    ax1.set_xlabel("Time")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.fill_between(df.index, df["drawdown"] * 100, 0.0, where=(df["drawdown"] < 0), alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(-100, 0)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels)

    plt.title("Balance (USDT) vs Drawdown (%)")
    plt.tight_layout()
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close()


def visualize_candle_and_position(
        symbols: List[str],
        timeframes: List[str],
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        position_data: pd.DataFrame,
        save_folder: str,
):
    for symbol in symbols:
        num = len(timeframes)
        fig = make_subplots(
            rows=num,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.0,
            row_heights=[0.5] * num,
            subplot_titles=timeframes,
        )

        position_df = position_data[position_data["symbol"] == symbol]
        for i, timeframe in enumerate(timeframes):
            df = data_dict[symbol][timeframe]
            df = truncate_dataframe(df, start_time=start_time, end_time=end_time)

            fig.add_trace(
                go.Candlestick(
                    x=df["timestamp"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    increasing_line_color="red",
                    decreasing_line_color="green",
                    name=timeframe,
                ),
                row=i+1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=position_df["buy_time"],
                    y=position_df["buy_price"],
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", color="magenta", size=12),
                    name="buy"
                ),
                row=i+1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=position_df["sell_time"],
                    y=position_df["sell_price"],
                    mode="markers+text",
                    marker=dict(symbol="triangle-down", color="royalblue", size=12),
                    name="sell"
                ),
                row=i+1,
                col=1,
            )

            fig.update_xaxes(rangeslider=dict(visible=False), row=i+1, col=1)

        fig.update_layout(
            title=symbol,
            hovermode="x unified",
            dragmode="pan",
        )

        os.makedirs(save_folder, exist_ok=True)
        fig.write_html(osp.join(save_folder, symbol.replace("/", "") + ".html"))
