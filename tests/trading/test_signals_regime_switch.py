import pandas as pd

from trading.signals import HybridTrendMeanRevStrategy, StrategyParams
from trading.types import LONG, FLAT


def test_trend_regime_entry_long():
    strategy = HybridTrendMeanRevStrategy(StrategyParams())
    row = pd.Series(
        {
            "ema_fast": 101,
            "ema_slow": 100,
            "macd_hist": 1.2,
            "rsi": 60,
            "bb_upper": 110,
            "bb_lower": 90,
            "atr": 1.0,
            "adx": 30,
            "close": 102,
        }
    )
    decision = strategy.decide(row, None)
    assert decision.target_side == LONG


def test_meanrev_regime_entry_flat_when_no_signal():
    strategy = HybridTrendMeanRevStrategy(StrategyParams())
    row = pd.Series(
        {
            "ema_fast": 101,
            "ema_slow": 100,
            "macd_hist": 1.2,
            "rsi": 50,
            "bb_upper": 110,
            "bb_lower": 90,
            "atr": 1.0,
            "adx": 10,
            "close": 100,
        }
    )
    decision = strategy.decide(row, None)
    assert decision.target_side == FLAT
