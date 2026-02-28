import pandas as pd

from trading.backtest import Backtester
from trading.config import TradingConfig
from trading.data_loader import SymbolFrames
from trading.portfolio import Portfolio
from trading.types import LONG


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
                "end_time": "20250101-010000",
            },
            "costs": {"fee_rate": 0.0, "slippage_bps": 0.0, "funding_rate_enabled": False},
            "risk": {"initial_balance": 1000, "max_gross_leverage": 2, "risk_per_trade": 0.01, "max_positions": 1},
            "strategy": {"name": "hybrid_trend_meanrev", "params": {}},
            "optimization": {"trials": 2},
            "output": {"save_folder": "results/trading"},
        }
    )


def test_stop_loss_priority_when_stop_and_take_hit_same_bar():
    ts = pd.to_datetime(["2025-01-01 00:01:00"])
    exec_df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0],
            "high": [120.0],
            "low": [80.0],
            "close": [100.0],
            "volume": [1.0],
        }
    )
    signal_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00"]),
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1.0],
        }
    )
    funding_df = pd.DataFrame({"timestamp": ts, "fundingRate": [0.0]})

    bt = Backtester(
        config=make_cfg(),
        market_store={"BTC/USDT": SymbolFrames(signal=signal_df, execution=exec_df, funding=funding_df)},
    )

    pf = Portfolio(initial_balance=1000.0, execution_model=bt.exec_model)
    pf.open_position(
        symbol="BTC/USDT",
        side=LONG,
        timestamp=pd.Timestamp("2025-01-01 00:00:00"),
        signal_time=pd.Timestamp("2025-01-01 00:00:00"),
        entry_price=100.0,
        qty=1.0,
        leverage=1.0,
        stop_price=90.0,
        take_price=110.0,
        reason="test",
        raw_price=100.0,
    )

    bt._check_stop_take(pd.Timestamp("2025-01-01 00:01:00"), pf)
    assert len(pf.trades) == 1
    assert pf.trades[0].exit_reason == "stop_loss"
