import pandas as pd

from trading.backtest import Backtester
from trading.config import TradingConfig
from trading.data_loader import SymbolFrames


def make_cfg():
    return TradingConfig.from_dict(
        {
            "engine": {"mode": "backtest", "seed": 42},
            "data": {
                "data_folder": "data",
                "symbols": ["BTC/USDT"],
                "timeframe_signal": "1h",
                "timeframe_execution": "1m",
                "start_time": "20250101-000000",
                "end_time": "20250101-030000",
            },
            "costs": {"fee_rate": 0.0, "slippage_bps": 0.0, "funding_rate_enabled": False},
            "risk": {"initial_balance": 1000, "max_gross_leverage": 2, "risk_per_trade": 0.01, "max_positions": 1},
            "strategy": {"name": "hybrid_trend_meanrev", "params": {}},
            "optimization": {"trials": 2},
            "output": {"save_folder": "results/trading"},
        }
    )


def test_next_exec_timestamp_is_strictly_after_signal_time():
    ts = pd.to_datetime(
        [
            "2025-01-01 00:00:00",
            "2025-01-01 00:01:00",
            "2025-01-01 00:02:00",
        ]
    )
    exec_df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    signal_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00"]),
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    funding_df = pd.DataFrame({"timestamp": ts, "fundingRate": [0.0, 0.0, 0.0]})

    bt = Backtester(
        config=make_cfg(),
        market_store={"BTC/USDT": SymbolFrames(signal=signal_df, execution=exec_df, funding=funding_df)},
    )

    nxt = bt._next_exec_timestamp("BTC/USDT", pd.Timestamp("2025-01-01 00:00:00"))
    assert nxt == pd.Timestamp("2025-01-01 00:01:00")
