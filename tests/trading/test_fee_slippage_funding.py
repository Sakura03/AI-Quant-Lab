import pandas as pd

from trading.execution import ExecutionModel
from trading.portfolio import Portfolio
from trading.types import LONG


def test_fee_slippage_and_funding_are_accounted():
    model = ExecutionModel(fee_rate=0.001, slippage_bps=10)
    pf = Portfolio(initial_balance=1000.0, execution_model=model)

    raw_entry = 100.0
    entry = model.apply_slippage(raw_entry, "buy")
    ok = pf.open_position(
        symbol="BTC/USDT",
        side=LONG,
        timestamp=pd.Timestamp("2025-01-01 00:01:00"),
        signal_time=pd.Timestamp("2025-01-01 00:00:00"),
        entry_price=entry,
        qty=1.0,
        leverage=1.0,
        stop_price=90.0,
        take_price=110.0,
        reason="unit_test",
        raw_price=raw_entry,
    )
    assert ok

    pf.apply_funding("BTC/USDT", funding_rate=0.001, mark_price=100.0)

    raw_exit = 101.0
    exit_px = model.apply_slippage(raw_exit, "sell")
    trade = pf.close_position("BTC/USDT", pd.Timestamp("2025-01-01 01:00:00"), exit_px, "exit", raw_exit)

    assert trade is not None
    assert trade.fees > 0
    assert trade.slippage_cost > 0
    assert trade.funding_pnl < 0
