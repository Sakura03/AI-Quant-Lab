import pandas as pd

from trading.optimize.walkforward import make_walk_forward_windows


def test_walkforward_windows_with_embargo_are_ordered_and_non_overlapping():
    windows = make_walk_forward_windows(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2022-01-01"),
        train_months=6,
        val_months=2,
        test_months=2,
        step_months=2,
        embargo_days=3,
    )
    assert windows
    for w in windows:
        assert w.train_start < w.train_end < w.val_start < w.val_end < w.test_start < w.test_end
        assert (w.val_start - w.train_end).days >= 3
        assert (w.test_start - w.val_end).days >= 3
