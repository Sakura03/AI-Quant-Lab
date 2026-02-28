import pandas as pd

from trading.data.catalog import timeframe_to_timedelta


def test_timestamp_is_close_time_semantics_for_open_time_derivation():
    close_time = pd.Timestamp("2025-01-01 10:00:00")
    open_time = close_time - timeframe_to_timedelta("1h")
    assert open_time == pd.Timestamp("2025-01-01 09:00:00")
