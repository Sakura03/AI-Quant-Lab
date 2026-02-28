import pandas as pd

from trading.features.engine import merge_regime_features


def test_merge_asof_uses_backward_closed_regime_bar_only():
    signal = pd.DataFrame(
        {
            "close_time": pd.to_datetime(["2025-01-01 10:00:00", "2025-01-01 10:30:00"]),
            "ema_fast": [1, 1],
            "ema_slow": [1, 1],
            "macd_hist": [0, 0],
            "rsi": [50, 50],
            "bb_mid": [100, 100],
            "atr": [1, 1],
            "adx": [20, 20],
            "ret_1": [0, 0],
            "ret_5": [0, 0],
            "vol_20": [0.1, 0.1],
            "zscore_20": [0.0, 0.0],
            "close": [100, 101],
        }
    )

    regime = pd.DataFrame(
        {
            "close_time": pd.to_datetime(["2025-01-01 09:00:00", "2025-01-01 11:00:00"]),
            "ema_fast": [10, 20],
            "ema_slow": [9, 21],
            "macd_hist": [1, -1],
            "rsi": [55, 45],
            "bb_mid": [100, 100],
            "atr": [2, 2],
            "adx": [25, 30],
            "ret_1": [0.01, -0.01],
            "ret_5": [0.02, -0.02],
            "vol_20": [0.2, 0.3],
            "zscore_20": [1.0, -1.0],
            "close": [100, 102],
        }
    )

    merged = merge_regime_features(signal, regime)

    # Both signal rows must map to 09:00 regime bar, not 11:00 future bar.
    assert merged.loc[0, "regime_ema_fast"] == 10
    assert merged.loc[1, "regime_ema_fast"] == 10
