import pandas as pd

from trading.metrics import compute_metrics


def test_metrics_have_drawdown_and_trade_count():
    eq = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="1min"),
            "equity": [100, 105, 103, 110, 108, 120],
            "cash": [100, 105, 103, 110, 108, 120],
            "open_positions": [0, 1, 1, 0, 0, 0],
            "gross_notional": [0, 10, 10, 0, 0, 0],
            "turnover": [0, 10, 10, 20, 20, 20],
        }
    )
    trades = pd.DataFrame({"pnl_usd": [2, -1, 3], "pnl_pct": [1.0, -0.5, 1.5]})
    m = compute_metrics(eq, trades, "1m")
    assert m.trade_count == 3
    assert m.max_drawdown >= 0
