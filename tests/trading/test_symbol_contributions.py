from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from trading.report.visualization import (
    build_symbol_contribution_frame,
    write_symbol_contribution_and_candles_html,
)


def test_build_symbol_contribution_frame_computes_cumulative_values():
    trades = pd.DataFrame(
        [
            {"symbol": "AAA", "pnl_usd": 10.0, "exit_time": "2025-01-01 00:01:00"},
            {"symbol": "AAA", "pnl_usd": -2.0, "exit_time": "2025-01-01 00:02:00"},
            {"symbol": "BBB", "pnl_usd": 5.0, "exit_time": "2025-01-01 00:02:00"},
            {"symbol": "AAA", "pnl_usd": 3.0, "exit_time": "2025-01-01 00:03:00"},
        ]
    )

    df = build_symbol_contribution_frame(trades)
    assert set(df["symbol"]) == {"AAA", "BBB"}

    aaa = df[df["symbol"] == "AAA"].sort_values("timestamp").reset_index(drop=True)
    assert list(aaa["contribution"]) == [10.0, 8.0, 11.0]

    bbb = df[df["symbol"] == "BBB"].sort_values("timestamp").reset_index(drop=True)
    assert list(bbb["contribution"]) == [5.0]


def test_build_symbol_contribution_frame_handles_empty_input():
    empty = pd.DataFrame(columns=["symbol", "pnl_usd", "exit_time"])
    df = build_symbol_contribution_frame(empty)
    assert df.empty
    assert list(df.columns) == ["timestamp", "symbol", "contribution"]


def test_write_symbol_contribution_and_candles_html_smoke(tmp_path):
    equity = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:01:00"]),
            "equity": [1000.0, 1010.0],
        }
    )
    bars = pd.DataFrame(
        {
            "close_time": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:01:00"]),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.5],
            "close": [100.5, 101.5],
            "volume": [1.0, 1.0],
        }
    )
    trades = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "entry_time": "2025-01-01 00:00:00",
                "exit_time": "2025-01-01 00:01:00",
                "entry_price": 100.0,
                "exit_price": 101.5,
                "pnl_usd": 1.5,
                "pnl_pct": 1.5,
                "side": "LONG",
            }
        ]
    )
    out_path = tmp_path / "combined.html"
    write_symbol_contribution_and_candles_html(
        equity, {"AAA": bars}, trades, "1m", str(out_path)
    )
    assert out_path.exists()


def test_write_symbol_contribution_and_candles_html_button_visibility(monkeypatch, tmp_path):
    captured: dict[str, go.Figure] = {}

    def _capture_write_html(self, file, *args, **kwargs):
        captured["fig"] = self
        Path(file).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(go.Figure, "write_html", _capture_write_html, raising=True)

    timestamps = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:01:00", "2025-01-01 00:02:00"])
    equity = pd.DataFrame({"timestamp": timestamps, "equity": [1000.0, 1005.0, 1002.0]})
    bars_aaa = pd.DataFrame(
        {
            "close_time": timestamps,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    bars_bbb = pd.DataFrame(
        {
            "close_time": timestamps,
            "open": [200.0, 201.0, 202.0],
            "high": [201.0, 202.0, 203.0],
            "low": [199.0, 200.0, 201.0],
            "close": [200.5, 201.5, 201.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    trades = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "entry_time": "2025-01-01 00:00:00",
                "exit_time": "2025-01-01 00:01:00",
                "entry_price": 100.0,
                "exit_price": 101.5,
                "pnl_usd": 1.5,
                "pnl_pct": 1.5,
                "side": "LONG",
            },
            {
                "symbol": "BBB",
                "entry_time": "2025-01-01 00:01:00",
                "exit_time": "2025-01-01 00:02:00",
                "entry_price": 201.5,
                "exit_price": 201.0,
                "pnl_usd": -0.5,
                "pnl_pct": -0.25,
                "side": "LONG",
            },
        ]
    )

    out_path = tmp_path / "combined_buttons.html"
    write_symbol_contribution_and_candles_html(
        equity,
        {"AAA": bars_aaa, "BBB": bars_bbb},
        trades,
        "1m",
        str(out_path),
    )
    assert out_path.exists()
    fig = captured["fig"]

    buttons = fig.layout.updatemenus[0].buttons
    assert [btn.label for btn in buttons] == ["AAA", "BBB"]

    names = [trace.name for trace in fig.data]
    equity_idx = names.index("Overall")
    aaa_candle_idx = names.index("AAA candles")
    bbb_candle_idx = names.index("BBB candles")
    assert "Trade Fills" in names
    assert "Entry" not in names and "Exit" not in names
    equity_trace = fig.data[equity_idx]
    assert float(equity_trace.y[0]) == 0.0
    assert float(max(equity_trace.y)) == 5.0

    vis_aaa = list(buttons[0].args[0]["visible"])
    vis_bbb = list(buttons[1].args[0]["visible"])
    assert vis_aaa[equity_idx]
    assert vis_bbb[equity_idx]
    assert vis_aaa[aaa_candle_idx] and not vis_aaa[bbb_candle_idx]
    assert vis_bbb[bbb_candle_idx] and not vis_bbb[aaa_candle_idx]

    ann_texts = [ann.text for ann in fig.layout.annotations]
    assert ann_texts.count("buy") == 2
    assert ann_texts.count("sell") == 2

    layout_aaa = buttons[0].args[1]
    ann_toggle_keys = [k for k in layout_aaa if k.startswith("annotations[")]
    assert ann_toggle_keys

    contrib_axis_range = list(fig.layout.yaxis.range)
    assert contrib_axis_range == [0.0, 5.0]
