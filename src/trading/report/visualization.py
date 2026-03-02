from __future__ import annotations

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


def build_symbol_contribution_frame(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative pnl per symbol keyed by exit timestamp."""
    columns = ["timestamp", "symbol", "contribution"]
    if trades_df.empty:
        return pd.DataFrame(columns=columns)

    trades = trades_df[["symbol", "exit_time", "pnl_usd"]].copy()
    trades["exit_time"] = _as_datetime(trades["exit_time"])
    trades = trades.dropna(subset=["symbol", "exit_time", "pnl_usd"])
    if trades.empty:
        return pd.DataFrame(columns=columns)

    frames: list[pd.DataFrame] = []
    grouped = trades.groupby(["symbol", "exit_time"], sort=True, as_index=False)["pnl_usd"].sum()
    for symbol, group in grouped.groupby("symbol", sort=True):
        symbol_df = group.sort_values("exit_time").reset_index(drop=True)
        symbol_df["cum_pnl"] = symbol_df["pnl_usd"].cumsum()
        symbol_df = symbol_df.rename(columns={"exit_time": "timestamp", "cum_pnl": "contribution"})
        frames.append(symbol_df[["timestamp", "symbol", "contribution"]])

    if not frames:
        return pd.DataFrame(columns=columns)

    return pd.concat(frames, ignore_index=True)


def write_symbol_contribution_html(
    equity_df: pd.DataFrame, contributions_df: pd.DataFrame, save_path: str
):
    """Plot stacked contributions with equity pnl (equity minus initial equity)."""
    if contributions_df.empty:
        return

    fig = go.Figure()
    equity = (
        equity_df.copy().sort_values("timestamp").reset_index(drop=True)
        if not equity_df.empty
        else pd.DataFrame()
    )
    equity["timestamp"] = _as_datetime(equity.get("timestamp", pd.Series(dtype="datetime64[ns]")))
    equity["equity"] = equity.get("equity", pd.Series(dtype=float)).astype(float)

    timestamps = sorted(
        set(contributions_df["timestamp"].dropna())
        | set(equity["timestamp"].dropna().tolist())
    )
    if not timestamps:
        return

    base_ts = pd.DataFrame({"timestamp": timestamps})
    for symbol in sorted(contributions_df["symbol"].unique()):
        sym_df = contributions_df[contributions_df["symbol"] == symbol]
        merged = base_ts.merge(sym_df, on="timestamp", how="left")
        merged["contribution"] = merged["contribution"].ffill().fillna(0.0)
        fig.add_trace(
            go.Scatter(
                x=merged["timestamp"],
                y=merged["contribution"],
                mode="lines",
                name=symbol,
                stackgroup="symbol_contributions",
                line=dict(width=0.0),
                hovertemplate="time=%{x}<br>%{y:.2f} USD<extra></extra>",
            )
        )

    initial_equity = float(equity["equity"].iloc[0]) if not equity.empty else 0.0
    max_equity = float(equity["equity"].max()) if not equity.empty else 1.0

    if not equity.empty:
        equity_pnl = equity["equity"] - initial_equity
        fig.add_trace(
            go.Scatter(
                x=equity["timestamp"],
                y=equity_pnl,
                mode="lines",
                name="Overall",
                line=dict(color="black", width=2),
                hovertemplate="time=%{x}<br>equity_pnl=%{y:.2f} USD<extra></extra>",
            )
        )

    fig.update_layout(
        title="Equity Contributions by Symbol",
        template="plotly_white",
        hovermode="x unified",
        yaxis=dict(
            title="Cumulative PnL (USD)",
            range=[0.0, max_equity - initial_equity],
        ),
        xaxis=dict(title="Time"),
    )
    fig.write_html(save_path, include_plotlyjs="cdn", full_html=True)


def write_symbol_contribution_and_candles_html(
    equity_df: pd.DataFrame,
    symbol_execution: dict[str, pd.DataFrame],
    trades_df: pd.DataFrame,
    candle_timeframe: str | None,
    save_path: str,
):
    """Render contribution and equity pnl curves plus synchronized symbol candles."""
    contributions_df = build_symbol_contribution_frame(trades_df)
    contribution_symbols = set(contributions_df["symbol"].unique())
    symbols = sorted(
        [
            s
            for s, bars in symbol_execution.items()
            if not bars.empty and s in contribution_symbols
        ]
    )
    if not symbols or contributions_df.empty:
        return

    equity = (
        equity_df.copy().sort_values("timestamp").reset_index(drop=True)
        if not equity_df.empty
        else pd.DataFrame()
    )
    equity["timestamp"] = _as_datetime(equity.get("timestamp", pd.Series(dtype="datetime64[ns]")))
    equity["equity"] = equity.get("equity", pd.Series(dtype=float)).astype(float)
    trades = trades_df.copy()
    if not trades.empty:
        for col in ["entry_time", "exit_time", "signal_time"]:
            if col in trades.columns:
                trades[col] = pd.to_datetime(trades[col], errors="coerce")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.5],
    )

    timestamps = sorted(
        set(contributions_df["timestamp"].dropna())
        | set(equity["timestamp"].dropna().tolist())
    )
    base_ts = pd.DataFrame({"timestamp": timestamps}) if timestamps else pd.DataFrame({"timestamp": []})
    initial_equity = float(equity["equity"].iloc[0]) if not equity.empty else 0.0
    max_equity = float(equity["equity"].max()) if not equity.empty else 1.0

    contrib_indices: list[int] = []
    for symbol in symbols:
        sym_df = contributions_df[contributions_df["symbol"] == symbol].sort_values("timestamp")
        merged = base_ts.merge(sym_df, on="timestamp", how="left")
        merged["contribution"] = merged["contribution"].ffill().fillna(0.0)
        trace = go.Scatter(
            x=merged["timestamp"],
            y=merged["contribution"],
            mode="lines",
            name=symbol,
            legendgroup=symbol,
            hovertemplate="time=%{x}<br>contribution=%{y:.2f} USD<extra></extra>",
        )
        fig.add_trace(trace, row=1, col=1)
        contrib_indices.append(len(fig.data) - 1)

    equity_idx: int | None = None
    if not equity.empty:
        equity_pnl = equity["equity"] - initial_equity
        fig.add_trace(
            go.Scatter(
                x=equity["timestamp"],
                y=equity_pnl,
                mode="lines",
                name="Overall",
                line=dict(color="black", width=2),
                hovertemplate="time=%{x}<br>equity_pnl=%{y:.2f} USD<extra></extra>",
            ),
            row=1,
            col=1,
        )
        equity_idx = len(fig.data) - 1

    fig.update_yaxes(
        title_text="Cumulative PnL (USD)",
        row=1,
        col=1,
        range=[0.0, max_equity - initial_equity],
    )

    symbol_trace_indices: dict[str, list[int]] = {}
    symbol_annotation_indices: dict[str, list[int]] = {}
    for idx, symbol in enumerate(symbols):
        bars = symbol_execution[symbol].copy().sort_values("close_time").reset_index(drop=True)
        bars["close_time"] = _as_datetime(bars["close_time"])
        candle_df = _resample_bars_for_plot(bars, candle_timeframe)
        if candle_df.empty:
            candle_df = bars
        visible = idx == 0
        start_idx = len(fig.data)
        candle_trace = go.Candlestick(
            x=candle_df["close_time"],
            open=candle_df["open"],
            high=candle_df["high"],
            low=candle_df["low"],
            close=candle_df["close"],
            name=f"{symbol} candles",
            legendgroup=symbol,
            visible=visible,
        )
        fig.add_trace(candle_trace, row=2, col=1)

        symbol_trades = trades[trades["symbol"] == symbol].copy()
        events = _build_trade_events(symbol_trades, symbol)
        ann_start = len(fig.layout.annotations) if fig.layout.annotations else 0
        _add_trade_markers_and_arrows(
            fig,
            events,
            symbol_trades,
            row=2,
            col=1,
            visible=visible,
        )
        ann_end = len(fig.layout.annotations) if fig.layout.annotations else 0
        symbol_annotation_indices[symbol] = list(range(ann_start, ann_end))
        symbol_trace_indices[symbol] = list(range(start_idx, len(fig.data)))
    total_traces = len(fig.data)
    base_visible = [False] * total_traces
    for idx in contrib_indices:
        base_visible[idx] = True
    if equity_idx is not None:
        base_visible[equity_idx] = True

    buttons = []
    total_annotations = len(fig.layout.annotations) if fig.layout.annotations else 0
    for symbol in symbols:
        visibility = base_visible.copy()
        for trace_idx in symbol_trace_indices.get(symbol, []):
            visibility[trace_idx] = True
        layout_update: dict[str, object] = {"title": f"Equity & Contributions for {symbol}"}
        if total_annotations:
            for ann_idx in range(total_annotations):
                layout_update[f"annotations[{ann_idx}].visible"] = False
            for ann_idx in symbol_annotation_indices.get(symbol, []):
                layout_update[f"annotations[{ann_idx}].visible"] = True
        buttons.append(
            dict(
                method="update",
                label=symbol,
                args=[
                    {"visible": visibility},
                    layout_update,
                ],
            )
        )

    fig.update_layout(
        title=f"Equity & Symbol Contributions | {symbols[0]}",
        template="plotly_white",
        hovermode="x unified",
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                showactive=True,
                buttons=buttons,
                x=0.0,
                y=1.12,
                xanchor="left",
                yanchor="top",
            )
        ],
    )
    fig.update_xaxes(title_text="Time", row=2, col=1, rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.write_html(save_path, include_plotlyjs="cdn", full_html=True)


def _add_trade_markers_and_arrows(
    fig: go.Figure,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    row: int | None = None,
    col: int = 1,
    visible: bool = True,
):
    """Overlay trade fills, buy/sell arrow annotations, trade paths and pnl text."""
    if events.empty and trades.empty:
        return

    def _add_trace(trace):
        if row is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig.add_trace(trace)

    def _add_annotation(*args, **kwargs):
        if row is not None:
            fig.add_annotation(row=row, col=col, *args, **kwargs)
        else:
            fig.add_annotation(*args, **kwargs)

    if not events.empty:
        clean_events = events.copy()
        clean_events["time"] = _as_datetime(clean_events["time"])
        clean_events["price"] = pd.to_numeric(clean_events["price"], errors="coerce")
        clean_events = clean_events.dropna(subset=["time", "price", "event", "action"])
        if not clean_events.empty:
            _add_trace(
                go.Scatter(
                    x=clean_events["time"],
                    y=clean_events["price"],
                    mode="markers",
                    name="Trade Fills",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color="black",
                        line=dict(color="black", width=1),
                    ),
                    customdata=clean_events["action"],
                    hovertemplate=(
                        "time=%{x}<br>"
                        + "price=%{y:.6f}<br>"
                        + "action=%{customdata}<extra></extra>"
                    ),
                    showlegend=False,
                    visible=visible,
                )
            )

            for event_row in clean_events.itertuples(index=False):
                action = str(event_row.action).lower()
                ay = 48 if action == "buy" else -48
                _add_annotation(
                    x=event_row.time,
                    y=float(event_row.price),
                    text=action,
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
                    visible=visible,
                )

    trades = trades.copy()
    if trades.empty:
        return

    def _to_float(value) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(number):
            return None
        return number

    line_x: list[object] = []
    line_y: list[float | None] = []
    pnl_x: list[pd.Timestamp] = []
    pnl_y: list[float] = []
    pnl_text: list[str] = []
    pnl_pos: list[str] = []

    for trade in trades.itertuples(index=False):
        if pd.isna(trade.entry_time) or pd.isna(trade.exit_time):
            continue
        entry_time = pd.Timestamp(trade.entry_time)
        exit_time = pd.Timestamp(trade.exit_time)
        entry_price = _to_float(getattr(trade, "entry_price", None))
        exit_price = _to_float(getattr(trade, "exit_price", None))
        if entry_price is None or exit_price is None:
            continue

        line_x.extend([entry_time, exit_time, None])
        line_y.extend([entry_price, exit_price, None])

        pnl_pct = getattr(trade, "pnl_pct", None)
        pnl_pct_float = _to_float(pnl_pct)
        if pnl_pct_float is None:
            continue
        pnl_x.append(exit_time)
        pnl_y.append(exit_price)
        pnl_text.append(f"{pnl_pct_float:+.2f}%")
        pnl_pos.append("top center" if pnl_pct_float >= 0 else "bottom center")

    if line_x:
        _add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode="lines",
                name="Trade Path",
                line=dict(color="gray", width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
                visible=visible,
            )
        )

    if pnl_x:
        _add_trace(
            go.Scatter(
                x=pnl_x,
                y=pnl_y,
                mode="text",
                text=pnl_text,
                textposition=pnl_pos,
                textfont=dict(size=10, color="black"),
                name="PnL %",
                hovertemplate="time=%{x}<br>pnl=%{text}<extra></extra>",
                showlegend=False,
                visible=visible,
            )
        )
