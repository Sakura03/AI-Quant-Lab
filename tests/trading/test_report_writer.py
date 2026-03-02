import pandas as pd

from trading.report.writer import _symbol_performance_summary


def test_symbol_performance_summary_aggregates_per_symbol():
    trades = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "pnl_usd": 10.0,
                "pnl_pct": 5.0,
                "holding_minutes": 30.0,
                "fees": 1.0,
                "slippage_cost": 0.5,
            },
            {
                "symbol": "AAA",
                "pnl_usd": -5.0,
                "pnl_pct": -2.0,
                "holding_minutes": 20.0,
                "fees": 1.0,
                "slippage_cost": 0.3,
            },
            {
                "symbol": "BBB",
                "pnl_usd": 15.0,
                "pnl_pct": 10.0,
                "holding_minutes": 60.0,
                "fees": 2.0,
                "slippage_cost": 1.0,
            },
        ]
    )

    summary = _symbol_performance_summary(trades)

    assert len(summary) == 2
    aaa = summary[summary["symbol"] == "AAA"].iloc[0]
    assert aaa["trade_count"] == 2
    assert aaa["win_rate"] == 0.5
    assert aaa["profit_factor"] == 2.0
    assert aaa["total_pnl_usd"] == 5.0
    assert aaa["avg_pnl_pct"] == 1.5
    assert aaa["avg_win_pct"] == 5.0
    assert aaa["avg_loss_pct"] == -2.0
    assert aaa["avg_holding_minutes"] == 25.0
    assert aaa["total_fees"] == 2.0
    assert aaa["avg_slippage_cost"] == 0.4

    bbb = summary[summary["symbol"] == "BBB"].iloc[0]
    assert bbb["trade_count"] == 1
    assert bbb["win_rate"] == 1.0
    assert bbb["profit_factor"] == float("inf")
    assert bbb["avg_pnl_pct"] == 10.0
    assert bbb["avg_win_pct"] == 10.0
    assert bbb["avg_loss_pct"] == 0.0
    assert bbb["avg_holding_minutes"] == 60.0
    assert bbb["total_fees"] == 2.0


def test_symbol_performance_summary_returns_schema_for_empty_df():
    empty = pd.DataFrame(columns=["symbol", "pnl_usd", "pnl_pct", "holding_minutes", "fees", "slippage_cost"])
    summary = _symbol_performance_summary(empty)
    expected_columns = [
        "symbol",
        "trade_count",
        "win_rate",
        "profit_factor",
        "total_pnl_usd",
        "avg_pnl_pct",
        "avg_win_pct",
        "avg_loss_pct",
        "avg_holding_minutes",
        "total_fees",
        "avg_slippage_cost",
    ]
    assert list(summary.columns) == expected_columns
    assert summary.empty
