from typing import List, Dict

import os
import os.path as osp
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import truncate_dataframe, merge_timeframes
from .indicator import add_indicator


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
    if len(df) == 0:
        return

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

        position_df = position_data[position_data["symbol"] == symbol].copy()
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

            price_pct = (position_df["sell_price"] / position_df["buy_price"] - 1.0) * 100

            buy_text = np.where(
                position_df["position_side"] == "Long",
                "B",
                price_pct.map(lambda x: f"B ({x:+.2f}%)"),
            )

            sell_text = np.where(
                position_df["position_side"] == "Long",
                price_pct.map(lambda x: f"S ({x:+.2f}%)"),
                "S",
            )

            buy_text = buy_text.tolist()
            sell_text = sell_text.tolist()

            # 买入点
            fig.add_trace(
                go.Scatter(
                    x=position_df["buy_time"],
                    y=position_df["buy_price"],
                    mode="markers+text",
                    marker=dict(
                        symbol="circle",
                        color="black",
                        size=8,
                    ),
                    text=buy_text,
                    textposition="bottom right",
                    showlegend=False,
                ),
                row=i+1,
                col=1,
            )

            # 卖出点
            fig.add_trace(
                go.Scatter(
                    x=position_df["sell_time"],
                    y=position_df["sell_price"],
                    mode="markers+text",
                    marker=dict(
                        symbol="circle",
                        color="black",
                        size=8,
                    ),
                    text=sell_text,
                    textposition="top right",
                    showlegend=False,
                ),
                row=i+1,
                col=1,
            )

            # 连线
            for _, row_trade in position_df.iterrows():
                fig.add_trace(
                    go.Scatter(
                        x=[row_trade["buy_time"], row_trade["sell_time"]],
                        y=[row_trade["buy_price"], row_trade["sell_price"]],
                        mode="lines",
                        line=dict(color="gray", width=1, dash="dash"),
                        showlegend=False,
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


def visualize_ichimoku(
        symbols: List[str],
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        position_data: pd.DataFrame,
        save_folder: str,
        indicator_params: Dict[str, int] = {},
):
    timeframe = "4h"
    for symbol in symbols:
        fig = go.Figure()

        df = data_dict[symbol][timeframe].copy()
        position_df = position_data[position_data["symbol"] == symbol]

        for indicator in indicator_params["4h"]:
            if indicator["name"] == "heikinashi":
                add_indicator(df, indicator)

        df_1d = copy.deepcopy(data_dict[symbol]["1d"])
        for indicator in indicator_params["1d"]:
            if indicator["name"] == "ichimoku":
                add_indicator(df_1d, indicator)

        df = merge_timeframes(df, df_1d, suffix="_1d")
        df = truncate_dataframe(df, start_time=start_time, end_time=end_time)

        ha_close_4h = df["ha_close"]
        senkou_a_1d = df["SenkouSpanA_1d"] if "SenkouSpanA_1d" in df.columns else df["SenkouSpanA"]
        senkou_b_1d = df["SenkouSpanB_1d"] if "SenkouSpanB_1d" in df.columns else df["SenkouSpanB"]

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
        )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=senkou_a_1d,
                mode="lines",
                line=dict(color="green", width=2),
                name="Senkou Span A",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=senkou_b_1d,
                mode="lines",
                line=dict(color="red", width=2),
                name="Senkou Span B",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=ha_close_4h,
                mode="lines",
                line=dict(color="blue", width=2),
                name="Heikin Ashi Close",
            ),
        )

        price_pct = (position_df["sell_price"] / position_df["buy_price"] - 1.0) * 100

        buy_text = np.where(
            position_df["position_side"] == "Long",
            "B",
            price_pct.map(lambda x: f"B ({x:+.2f}%)"),
        )

        sell_text = np.where(
            position_df["position_side"] == "Long",
            price_pct.map(lambda x: f"S ({x:+.2f}%)"),
            "S",
        )

        buy_text = buy_text.tolist()
        sell_text = sell_text.tolist()

        # 买入点
        fig.add_trace(
            go.Scatter(
                x=position_df["buy_time"],
                y=position_df["buy_price"],
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    color="black",
                    size=8,
                ),
                text=buy_text,
                textposition="bottom right",
                showlegend=False,
            ),
        )

        # 卖出点
        fig.add_trace(
            go.Scatter(
                x=position_df["sell_time"],
                y=position_df["sell_price"],
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    color="black",
                    size=8,
                ),
                text=sell_text,
                textposition="top right",
                showlegend=False,
            ),
        )

        # 连线
        for _, row_trade in position_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row_trade["buy_time"], row_trade["sell_time"]],
                    y=[row_trade["buy_price"], row_trade["sell_price"]],
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dash"),
                    showlegend=False,
                ),
            )

        fig.update_xaxes(rangeslider=dict(visible=False))
        fig.update_layout(
            title=symbol + " " + timeframe,
            hovermode="x unified",
            dragmode="pan",
        )

        os.makedirs(save_folder, exist_ok=True)
        fig.write_html(osp.join(save_folder, symbol.replace("/", "") + "_ichimoku.html"))
