import pandas as pd

from trading.optimizer import make_walk_forward_windows


def test_walk_forward_windows_are_ordered_and_non_leaking():
    windows = make_walk_forward_windows(
        start_time=pd.Timestamp("2025-01-01"),
        end_time=pd.Timestamp("2026-01-01"),
        train_months=6,
        valid_months=2,
        test_months=2,
    )

    assert len(windows) >= 1
    for w in windows:
        assert w["train_start"] < w["train_end"] <= w["valid_start"] < w["valid_end"] <= w["test_start"] < w["test_end"]
