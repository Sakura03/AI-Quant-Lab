import pandas as pd
import pytest

from trading.config import DataConfig
from trading.data.loader import SymbolDataBundle, load_market_bundles


class _StubLoader:
    def __init__(self, fail_symbols):
        self.fail_symbols = set(fail_symbols)

    def build_bundle(self, symbol, signal_tf, regime_tf, start, end, warmup_bars):
        if symbol in self.fail_symbols:
            raise ValueError(f"Execution frame is empty for {symbol}")
        ts = pd.to_datetime(["2025-01-01 00:00:00"])
        bars = pd.DataFrame(
            {
                "close_time": ts,
                "open_time": ts - pd.Timedelta(minutes=1),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
                "timestamp": ts,
                "timeframe": ["1m"],
            }
        )
        funding = pd.DataFrame({"close_time": ts, "fundingRate": [0.0]})
        return SymbolDataBundle(signal=bars, regime=bars, execution=bars, funding=funding)


def _cfg(symbols):
    return DataConfig.from_dict(
        {
            "root": "data",
            "universe": symbols,
            "exec_tf": "1m",
            "signal_tfs": ["1h"],
            "regime_tfs": ["4h"],
            "start": "20250101-000000",
            "end": "20250201-000000",
            "warmup_bars": 10,
            "timestamp_semantics": "close",
        }
    )


def test_loader_skips_symbols_with_empty_execution_data():
    out = load_market_bundles(
        data_cfg=_cfg(["BTC/USDT", "DOGE/USDT"]),
        signal_tf="1h",
        regime_tf="4h",
        start=pd.Timestamp("2025-01-01"),
        end=pd.Timestamp("2025-02-01"),
        warmup_bars=10,
        loader=_StubLoader(fail_symbols=["DOGE/USDT"]),
    )
    assert "BTC/USDT" in out
    assert "DOGE/USDT" not in out


def test_loader_raises_if_all_symbols_empty():
    with pytest.raises(ValueError, match="No symbols have valid execution data"):
        load_market_bundles(
            data_cfg=_cfg(["DOGE/USDT"]),
            signal_tf="1h",
            regime_tf="4h",
            start=pd.Timestamp("2025-01-01"),
            end=pd.Timestamp("2025-02-01"),
            warmup_bars=10,
            loader=_StubLoader(fail_symbols=["DOGE/USDT"]),
        )
