import pandas as pd

from trading.backtest.engine import BacktestEngine
from trading.config import DataConfig, TradingConfig
from trading.data.loader import MarketDataLoader, SymbolDataBundle


def _write_hourly_ohlcv(path: str):
    ts = pd.to_datetime([
        "2025-01-01 00:00:00",
        "2025-01-01 01:00:00",
        "2025-01-01 02:00:00",
    ])
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )
    df.to_feather(path)


def test_loader_open_timestamp_semantics_filters_by_open_time(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _write_hourly_ohlcv(str(root / "BTC_USDT-1h.feather"))

    cfg = DataConfig.from_dict(
        {
            "root": str(root),
            "universe": ["BTC/USDT"],
            "exec_tf": "1h",
            "signal_tfs": ["1h"],
            "regime_tfs": ["1h"],
            "start": "20250101-010000",
            "end": "20250101-020000",
            "warmup_bars": 0,
            "timestamp_semantics": "open",
        }
    )

    loader = MarketDataLoader(cfg)
    out = loader.get_execution_bars("BTC/USDT", pd.Timestamp("2025-01-01 01:00:00"), pd.Timestamp("2025-01-01 02:00:00"))

    assert out["open_time"].tolist() == [pd.Timestamp("2025-01-01 01:00:00"), pd.Timestamp("2025-01-01 02:00:00")]
    assert out["close_time"].tolist() == [pd.Timestamp("2025-01-01 02:00:00"), pd.Timestamp("2025-01-01 03:00:00")]


def _engine_cfg() -> TradingConfig:
    return TradingConfig.from_dict(
        {
            "engine": {"seed": 1, "n_jobs": 1},
            "data": {
                "root": "data",
                "universe": ["BTC/USDT"],
                "exec_tf": "1h",
                "signal_tfs": ["1h"],
                "regime_tfs": ["1h"],
                "start": "20250101-010000",
                "end": "20250101-020000",
                "warmup_bars": 0,
                "timestamp_semantics": "open",
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


def test_backtest_timeline_uses_open_time_range_for_open_semantics():
    exec_df = pd.DataFrame(
        {
            "open_time": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00", "2025-01-01 02:00:00"]),
            "close_time": pd.to_datetime(["2025-01-01 01:00:00", "2025-01-01 02:00:00", "2025-01-01 03:00:00"]),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
            "timeframe": ["1h", "1h", "1h"],
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00", "2025-01-01 02:00:00"]),
        }
    )

    signal = exec_df.copy()
    regime = exec_df.copy()
    funding = pd.DataFrame({"close_time": pd.to_datetime([]), "fundingRate": pd.Series(dtype=float)})

    bt = BacktestEngine(
        config=_engine_cfg(),
        signal_tf="1h",
        regime_tf="1h",
        strategy_ids=["trend_following"],
        strategy_params={"trend_following": {}},
        market_bundles={"BTC/USDT": SymbolDataBundle(signal=signal, regime=regime, execution=exec_df, funding=funding)},
    )

    timeline = bt._build_timeline(pd.Timestamp("2025-01-01 01:00:00"), pd.Timestamp("2025-01-01 02:00:00"))
    assert timeline == [pd.Timestamp("2025-01-01 02:00:00"), pd.Timestamp("2025-01-01 03:00:00")]
