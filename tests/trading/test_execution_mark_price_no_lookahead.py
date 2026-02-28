import pandas as pd

from trading.backtest.engine import BacktestEngine
from trading.config import TradingConfig
from trading.domain.types import LONG
from trading.data.loader import SymbolDataBundle
from trading.portfolio.allocator import PortfolioLedger
from trading.strategies.base import StrategyDecision


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


def test_execute_due_orders_uses_previous_marks_not_current_bar_close():
    exec_df = pd.DataFrame(
        {
            "close_time": pd.to_datetime(["2025-01-01 00:01:00", "2025-01-01 00:02:00"]),
            "open_time": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:01:00"]),
            "open": [100.0, 100.0],
            "high": [100.0, 200.0],
            "low": [100.0, 100.0],
            "close": [100.0, 200.0],
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

    seen_marks: list[tuple[pd.Timestamp, dict[str, float]]] = []
    original = bt._execute_due_orders

    def _capture(ts, queue, ledger, mark_prices):
        seen_marks.append((ts, dict(mark_prices)))
        return original(ts, queue, ledger, mark_prices)

    bt._execute_due_orders = _capture
    bt.run()

    assert len(seen_marks) == 2
    assert seen_marks[0][1] == {}
    assert seen_marks[1][1]["BTC/USDT"] == 100.0


def test_open_order_sizing_does_not_use_next_bar_open_price():
    exec_df = pd.DataFrame(
        {
            "close_time": pd.to_datetime(["2025-01-01 00:01:00"]),
            "open_time": pd.to_datetime(["2025-01-01 00:00:00"]),
            "open": [1000.0],  # intentionally far from signal mark price
            "high": [1000.0],
            "low": [1000.0],
            "close": [1000.0],
            "volume": [1.0],
            "timestamp": pd.to_datetime(["2025-01-01 00:01:00"]),
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
    order = bt._build_open_order(
        symbol="BTC/USDT",
        target_side=LONG,
        strategy_id="trend_following",
        decision=StrategyDecision(
            side=LONG,
            strength=1.0,
            stop_atr=2.0,
            take_atr=3.0,
            confidence=0.7,
            reason="test",
        ),
        signal_time=pd.Timestamp("2025-01-01 00:00:00"),
        next_exec_close_time=pd.Timestamp("2025-01-01 00:01:00"),
        feature_row=pd.Series({"atr": 1.0}),
        ledger=ledger,
        mark_prices={"BTC/USDT": 100.0},
    )

    assert order is not None
    # If sizing used next bar open=1000, qty would be ~10. Expected uses signal mark=100 => qty ~50.
    assert order.qty > 40.0
