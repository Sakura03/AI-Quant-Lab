from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class WalkForwardWindow:
    """One walk-forward split window with train/val/test boundaries."""

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    """Shift timestamp forward by N calendar months."""
    return ts + pd.DateOffset(months=months)


def make_walk_forward_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    val_months: int,
    test_months: int,
    step_months: int,
    embargo_days: int,
) -> list[WalkForwardWindow]:
    """Generate sequential walk-forward windows with optional embargo gaps."""
    windows: list[WalkForwardWindow] = []
    cursor = pd.Timestamp(start)
    embargo = pd.Timedelta(days=max(0, embargo_days))
    wid = 1

    while True:
        train_start = cursor
        train_end = add_months(train_start, train_months)
        val_start = train_end + embargo
        val_end = add_months(val_start, val_months)
        test_start = val_end + embargo
        test_end = add_months(test_start, test_months)

        if test_end > end:
            break

        windows.append(
            WalkForwardWindow(
                window_id=wid,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        cursor = add_months(cursor, step_months)
        wid += 1

    return windows
