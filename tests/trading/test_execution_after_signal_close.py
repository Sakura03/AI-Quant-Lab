import pandas as pd

from trading.backtest.engine import BacktestEngine
from trading.config import TradingConfig
from trading.data.loader import SymbolDataBundle


def _cfg() -> TradingConfig:
    return TradingConfig.from_dict(
        {
            "engine": {"seed": 1, "n_jobs": 1},
            "data": {
                "root": "data",
                "universe": ["BTC/USDT"],
                "exec_tf": "1m",
                "signal_tfs": ["1h"],
                "regime_tfs": ["1h"],
                "start": "20250101-000000",
                "end": "20250101-010000",
                "warmup_bars": 10,
                "timestamp_semantics": "close",
            },
            "split": {"train_months": 1, "val_months": 1, "test_months": 1, "step_months": 1, "embargo_days": 0},
            "strategies": {"enabled": ["trend_following"], "per_strategy_max_positions": 1},
            "portfolio": {
                "initial_balance": 10000,
                "max_gross_leverage": 1.5,
                "max_total_positions": 1,
                "max_symbol_weight": 1.0,
                "target_vol_annual": 0.2,
                "dd_soft": 0.08,
                "dd_hard": 0.2,
                "min_risk_scale": 0.2,
                "risk_per_trade": 0.01,
            },
            "costs": {"fee_rate": 0.0, "slippage_bps": 0.0, "funding_enabled": False},
            "optimizer": {
                "trials_per_window": 10,
                "population": 4,
                "elite_top_k": 2,
                "prior_adoption_prob": 0.7,
                "mutation_strength": 0.2,
                "crossover_prob": 0.3,
                "min_trades_per_window": 1,
            },
            "objective": {"weights": {}, "hard_limits": {"max_drawdown": 0.5}},
            "output": {"dir": "results/trading_enterprise"},
        }
    )


def test_next_exec_close_time_is_from_first_bar_with_open_after_signal_close():
    exec_df = pd.DataFrame(
        {
            "close_time": pd.to_datetime(["2025-01-01 00:00:30", "2025-01-01 00:01:00", "2025-01-01 00:02:00"]),
            "open_time": pd.to_datetime(["2024-12-31 23:59:30", "2025-01-01 00:00:00", "2025-01-01 00:01:00"]),
            "open": [100.0, 100.1, 100.2],
            "high": [100.0, 100.1, 100.2],
            "low": [100.0, 100.1, 100.2],
            "close": [100.0, 100.1, 100.2],
            "volume": [1.0, 1.0, 1.0],
            "timestamp": pd.to_datetime(["2025-01-01 00:00:30", "2025-01-01 00:01:00", "2025-01-01 00:02:00"]),
            "timeframe": ["1m", "1m", "1m"],
        }
    )
    sig = pd.DataFrame(
        {
            "close_time": pd.to_datetime(["2025-01-01 00:00:00"]),
            "open_time": pd.to_datetime(["2024-12-31 23:00:00"]),
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1.0],
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00"]),
            "timeframe": ["1h"],
        }
    )
    funding = pd.DataFrame({"close_time": pd.to_datetime(["2025-01-01 00:01:00"]), "fundingRate": [0.0]})

    bt = BacktestEngine(
        config=_cfg(),
        signal_tf="1h",
        regime_tf="1h",
        strategy_ids=["trend_following"],
        strategy_params={"trend_following": {}},
        market_bundles={"BTC/USDT": SymbolDataBundle(signal=sig, regime=sig, execution=exec_df, funding=funding)},
    )

    nxt = bt._next_exec_close_time("BTC/USDT", pd.Timestamp("2025-01-01 00:00:00"))
    assert nxt == pd.Timestamp("2025-01-01 00:01:00")
