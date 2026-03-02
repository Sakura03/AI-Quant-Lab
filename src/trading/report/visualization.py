from __future__ import annotations

import os
import os.path as osp
import re

import matplotlib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _as_datetime(series: pd.Series) -> pd.Series:
    """Convert input series to pandas datetime."""
    if series.empty:
        return series
    return pd.to_datetime(series)


def write_equity_drawdown_html(equity_df: pd.DataFrame, save_path: str):
    """Render combined equity and drawdown as interactive HTML figure."""
    if equity_df.empty:
        return

    df = equity_df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = _as_datetime(df["timestamp"])
    equity = df["equity"].astype(float)
    if "drawdown" in df.columns:
        dd = df["drawdown"].astype(float)
    else:
        peak = equity.cummax()
        dd = 1.0 - equity / peak
    dd_pct = dd * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=equity,
            mode="lines",
            name="Equity",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="time=%{x}<br>equity=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=dd_pct,
            mode="lines",
            name="Drawdown %",
            line=dict(color="#d62728", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(214,39,40,0.2)",
            yaxis="y2",
            hovertemplate="time=%{x}<br>drawdown=%{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Backtest Equity & Drawdown",
        template="plotly_white",
        height=820,
        hovermode="x unified",
        yaxis=dict(title="Equity", side="left"),
        yaxis2=dict(
            title="Drawdown %",
            overlaying="y",
            side="right",
            range=[100, 0],
            showgrid=False,
            ticks="outside",
        ),
        xaxis=dict(title="Time"),
    )
    fig.write_html(save_path, include_plotlyjs="cdn", full_html=True)


def write_equity_drawdown_png(equity_df: pd.DataFrame, save_path: str):
    """Render equity and drawdown as static PNG image."""
    if equity_df.empty:
        return

    df = equity_df.copy().sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = _as_datetime(df["timestamp"])
    equity = df["equity"].astype(float)
    if "drawdown" in df.columns:
        dd = df["drawdown"].astype(float)
    else:
        dd = 1.0 - equity / equity.cummax()
    dd_pct = dd * 100.0

    fig, ax_eq = plt.subplots(figsize=(14, 6))
    ax_dd = ax_eq.twinx()

    ax_eq.plot(df["timestamp"], equity, color="#1f77b4", linewidth=1.8, label="Equity")
    ax_eq.set_ylabel("Equity", color="#1f77b4")
    ax_eq.tick_params(axis="y", colors="#1f77b4")
    ax_eq.grid(alpha=0.25)

    ax_dd.plot(df["timestamp"], dd_pct, color="#d62728", linewidth=1.4, label="Drawdown %")
    ax_dd.fill_between(df["timestamp"], dd_pct, 0.0, color="#d62728", alpha=0.2)
    ax_dd.set_ylabel("Drawdown %", color="#d62728")
    ax_dd.set_ylim(100, 0)
    ax_dd.tick_params(axis="y", colors="#d62728")
    ax_dd.set_xlabel("Time")

    fig.suptitle("Backtest Equity & Drawdown")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _safe_symbol(symbol: str) -> str:
    """Convert symbol string into filename-safe token."""
    return symbol.replace("/", "_")


def _parse_timeframe(timeframe: str) -> tuple[str, pd.Timedelta] | None:
    """Parse timeframe text into pandas resample rule and timedelta."""
    m = re.fullmatch(r"(\d+)([mhd])", str(timeframe).strip().lower())
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return f"{n}min", pd.Timedelta(minutes=n)
    if unit == "h":
        return f"{n}h", pd.Timedelta(hours=n)
    return f"{n}d", pd.Timedelta(days=n)


def _resample_bars_for_plot(bars: pd.DataFrame, target_timeframe: str | None) -> pd.DataFrame:
    """Resample raw execution bars to plotting timeframe when target is coarser."""
    if bars.empty or not target_timeframe:
        return bars

    parsed = _parse_timeframe(target_timeframe)
    if parsed is None:
        return bars
    rule, target_delta = parsed

    ts = bars["close_time"].sort_values().drop_duplicates()
    diffs = ts.diff().dropna()
    if diffs.empty:
        return bars
    source_delta = diffs.median()
    if pd.isna(source_delta) or source_delta <= pd.Timedelta(0):
        return bars
    if target_delta <= source_delta:
        return bars

    work = bars.copy().sort_values("close_time").reset_index(drop=True)
    work["close_time"] = _as_datetime(work["close_time"])
    ohlc = (
        work.set_index("close_time")
        .resample(rule, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    if ohlc.empty:
        return bars
    return ohlc


def _build_trade_events(trades_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Build point-level buy/sell events from entry/exit trades for one symbol."""
    if trades_df.empty or "symbol" not in trades_df.columns:
        return pd.DataFrame(columns=["time", "price", "action", "event"])

    rows = trades_df[trades_df["symbol"] == symbol].copy()
    if rows.empty:
        return pd.DataFrame(columns=["time", "price", "action", "event"])

    entry = rows[["entry_time", "entry_price", "side"]].copy()
    entry["event"] = "entry"
    entry["action"] = entry["side"].map(lambda s: "buy" if str(s).upper() == "LONG" else "sell")
    entry = entry.rename(columns={"entry_time": "time", "entry_price": "price"})

    exit_ = rows[["exit_time", "exit_price", "side"]].copy()
    exit_["event"] = "exit"
    exit_["action"] = exit_["side"].map(lambda s: "sell" if str(s).upper() == "LONG" else "buy")
    exit_ = exit_.rename(columns={"exit_time": "time", "exit_price": "price"})

    out = pd.concat(
        [
            entry[["time", "price", "action", "event"]],
            exit_[["time", "price", "action", "event"]],
        ],
        ignore_index=True,
    )
    out["time"] = _as_datetime(out["time"])
    out = out.sort_values("time").reset_index(drop=True)
    return out


def _add_trade_markers_and_arrows(fig: go.Figure, events: pd.DataFrame, trades: pd.DataFrame):
    """Overlay black fill markers and buy/sell arrow annotations."""
    if events.empty:
        return

    fig.add_trace(
        go.Scatter(
            x=events["time"],
            y=events["price"],
            mode="markers",
            name="Trade Fills",
            marker=dict(symbol="circle", size=8, color="black", line=dict(color="black", width=1)),
            hovertemplate="time=%{x}<br>price=%{y:.6f}<extra></extra>",
        )
    )

    for row in events.itertuples(index=False):
        ay = 48 if row.action == "buy" else -48
        fig.add_annotation(
            x=row.time,
            y=float(row.price),
            text=row.action,
            showarrow=True,
            arrowhead=2,
            arrowsize=1.1,
            arrowwidth=1.4,
            arrowcolor="black",
            ax=0,
            ay=ay,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="black",
            borderwidth=1,
            opacity=0.98,
        )

    trades = trades.copy()
    if trades.empty:
        return

    for trade in trades.itertuples(index=False):
        if pd.isna(trade.entry_time) or pd.isna(trade.exit_time):
            continue
        entry_time = pd.Timestamp(trade.entry_time)
        exit_time = pd.Timestamp(trade.exit_time)
        entry_price = float(getattr(trade, "entry_price", 0.0))
        exit_price = float(getattr(trade, "exit_price", 0.0))

        fig.add_trace(
            go.Scatter(
                x=[entry_time, exit_time],
                y=[entry_price, exit_price],
                mode="lines",
                name="Trade Path",
                line=dict(color="gray", width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        pnl_pct = getattr(trade, "pnl_pct", None)
        if pnl_pct is None or pd.isna(pnl_pct):
            continue
        pnl_text = f"{pnl_pct:+.2f}%"
        text_shift = 14 if pnl_pct >= 0 else -14
        fig.add_annotation(
            x=exit_time,
            y=exit_price,
            text=pnl_text,
            showarrow=False,
            yshift=text_shift,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
        )


def write_symbol_candles_with_trades_html(
    symbol_execution: dict[str, pd.DataFrame],
    trades_df: pd.DataFrame,
    out_dir: str,
    candle_timeframe: str | None = None,
):
    """Render per-symbol candlestick HTML with trade markers/annotations."""
    os.makedirs(out_dir, exist_ok=True)

    trades = trades_df.copy()
    if not trades.empty:
        for col in ["entry_time", "exit_time", "signal_time"]:
            if col in trades.columns:
                trades[col] = pd.to_datetime(trades[col])

    for symbol, bars in symbol_execution.items():
        if bars.empty:
            continue

        df = bars.copy().sort_values("close_time").reset_index(drop=True)
        df["close_time"] = _as_datetime(df["close_time"])
        candle_df = _resample_bars_for_plot(df, candle_timeframe)

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=candle_df["close_time"],
                open=candle_df["open"],
                high=candle_df["high"],
                low=candle_df["low"],
                close=candle_df["close"],
                name=symbol,
            )
        )

        symbol_trades = trades[trades["symbol"] == symbol].copy()
        events = _build_trade_events(symbol_trades, symbol)
        _add_trade_markers_and_arrows(fig, events, symbol_trades)

        tf_note = f" ({candle_timeframe})" if candle_timeframe else ""
        fig.update_layout(
            title=f"{symbol} Candles{tf_note} with Trade Fills",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_white",
            xaxis_rangeslider_visible=False,
            height=760,
            hovermode="x",
        )

        filename = f"candles_{_safe_symbol(symbol)}.html"
        fig.write_html(osp.join(out_dir, filename), include_plotlyjs="cdn", full_html=True)
