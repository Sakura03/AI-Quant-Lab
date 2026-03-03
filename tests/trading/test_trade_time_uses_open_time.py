import pandas as pd

from trading.backtest.engine import BacktestEngine, ScheduledOrder
from trading.config import TradingConfig
from trading.data.loader import SymbolDataBundle
from trading.domain.types import LONG
from trading.portfolio.allocator import PortfolioLedger


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
                "start": "20250101-000100",
                "end": "20250101-000200",
                "warmup_bars": 5,
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


def test_execute_due_orders_records_entry_and_exit_at_execution_bar_open_time():
    exec_df = pd.DataFrame(
        {
            "close_time": pd.to_datetime(["2025-01-01 00:01:00", "2025-01-01 00:02:00"]),
            "open_time": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:01:00"]),
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": [1.0, 1.0],
            "timestamp": pd.to_datetime(["2025-01-01 00:01:00", "2025-01-01 00:02:00"]),
            "timeframe": ["1m", "1m"],
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
    funding = pd.DataFrame({"close_time": pd.to_datetime([]), "fundingRate": pd.Series(dtype=float)})

    bt = BacktestEngine(
        config=_cfg(),
        signal_tf="1h",
        regime_tf="1h",
        strategy_ids=["trend_following"],
        strategy_params={"trend_following": {}},
        market_bundles={"BTC/USDT": SymbolDataBundle(signal=sig, regime=sig, execution=exec_df, funding=funding)},
    )

    ledger = PortfolioLedger(10000.0, bt.execution)
    queue: dict[pd.Timestamp, list[ScheduledOrder]] = {}

    entry_close_time = pd.Timestamp("2025-01-01 00:01:00")
    bt._enqueue(
        queue,
        ScheduledOrder(
            kind="open",
            symbol="BTC/USDT",
            execution_close_time=entry_close_time,
            signal_time=pd.Timestamp("2025-01-01 00:00:00"),
            side=LONG,
            qty=1.0,
            leverage=1.0,
            stop_price=99.0,
            take_price=102.0,
            strategy="trend_following",
            reason="test_entry",
        ),
    )
    bt._execute_due_orders(entry_close_time, queue, ledger, {})
    assert ledger.positions["BTC/USDT"].entry_time == pd.Timestamp("2025-01-01 00:00:00")

    exit_close_time = pd.Timestamp("2025-01-01 00:02:00")
    bt._enqueue(
        queue,
        ScheduledOrder(
            kind="close",
            symbol="BTC/USDT",
            execution_close_time=exit_close_time,
            signal_time=pd.Timestamp("2025-01-01 00:01:00"),
            strategy="trend_following",
            reason="test_exit",
        ),
    )
    bt._execute_due_orders(exit_close_time, queue, ledger, {"BTC/USDT": 100.0})

    assert len(ledger.trades) == 1
    assert ledger.trades[0].exit_time == pd.Timestamp("2025-01-01 00:01:00")


def test_check_stop_take_closes_position_at_execution_bar_close_time():
    exec_df = pd.DataFrame(
        {
            "close_time": pd.to_datetime(["2025-01-01 00:02:00"]),
            "open_time": pd.to_datetime(["2025-01-01 00:01:00"]),
            "open": [100.0],
            "high": [100.0],
            "low": [98.0],  # hit long stop at 99
            "close": [99.5],
            "volume": [1.0],
            "timestamp": pd.to_datetime(["2025-01-01 00:02:00"]),
            "timeframe": ["1m"],
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
    funding = pd.DataFrame({"close_time": pd.to_datetime([]), "fundingRate": pd.Series(dtype=float)})

    bt = BacktestEngine(
        config=_cfg(),
        signal_tf="1h",
        regime_tf="1h",
        strategy_ids=["trend_following"],
        strategy_params={"trend_following": {}},
        market_bundles={"BTC/USDT": SymbolDataBundle(signal=sig, regime=sig, execution=exec_df, funding=funding)},
    )

    ledger = PortfolioLedger(10000.0, bt.execution)
    opened = ledger.open_position(
        symbol="BTC/USDT",
        side=LONG,
        timestamp=pd.Timestamp("2025-01-01 00:01:00"),
        signal_time=pd.Timestamp("2025-01-01 00:00:00"),
        strategy="trend_following",
        entry_reason="test_entry",
        entry_price=100.0,
        raw_price=100.0,
        qty=1.0,
        leverage=1.0,
        stop_price=99.0,
        take_price=105.0,
    )
    assert opened

    bt._check_stop_take(pd.Timestamp("2025-01-01 00:02:00"), ledger)

    assert len(ledger.trades) == 1
    assert ledger.trades[0].exit_reason == "stop_loss"
    assert ledger.trades[0].exit_time == pd.Timestamp("2025-01-01 00:02:00")
