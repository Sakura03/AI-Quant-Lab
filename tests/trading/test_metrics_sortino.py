import math

import numpy as np
import pandas as pd

from trading.evaluate.metrics import compute_metrics


def _equity_from_returns(rets: list[float], start: float = 1000.0) -> pd.Series:
    values = [start]
    current = start
    for r in rets:
        current *= 1.0 + r
        values.append(current)
    return pd.Series(values, dtype=float)


def test_sortino_uses_zeroed_positive_returns_for_downside_std():
    raw_rets = [0.08, -0.03, 0.05, -0.02, 0.04, 0.01]
    equity = _equity_from_returns(raw_rets)
    timestamps = pd.date_range("2025-01-01", periods=len(equity), freq="D")
    equity_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "equity": equity,
            "open_positions": np.zeros(len(equity)),
            "turnover": np.zeros(len(equity)),
        }
    )

    metrics = compute_metrics(equity_df, pd.DataFrame(), execution_timeframe="1d")

    rets = equity_df["equity"].pct_change().dropna()
    mean_ret = float(rets.mean())
    downside = rets.clip(upper=0.0)
    downside_std = float(downside.std(ddof=0))
    expected = math.sqrt(365.0) * mean_ret / downside_std

    old_downside = rets[rets < 0]
    old_std = float(old_downside.std(ddof=0))
    old_formula = math.sqrt(365.0) * mean_ret / old_std

    assert math.isclose(metrics.sortino, expected, rel_tol=1e-12, abs_tol=1e-12)
    assert not math.isclose(metrics.sortino, old_formula, rel_tol=1e-6, abs_tol=1e-6)
