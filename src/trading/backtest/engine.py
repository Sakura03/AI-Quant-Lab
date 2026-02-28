from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from trading.config import TradingConfig
from trading.data.loader import MarketDataLoader, SymbolDataBundle, load_market_bundles
from trading.domain.types import BacktestResult, FLAT, LONG, SHORT
from trading.evaluate.metrics import compute_metrics
from trading.execution.simulator import ExecutionSimulator, side_to_close_order, side_to_open_order
from trading.features.engine import build_feature_frame
from trading.portfolio.allocator import PortfolioLedger
from trading.portfolio.risk_guards import drawdown_risk_scale
from trading.strategies.base import StrategyDecision
from trading.strategies.factory import build_strategy


@dataclass
class ScheduledOrder:
    kind: str  # open / close
    symbol: str
    execution_close_time: pd.Timestamp
    signal_time: pd.Timestamp
    side: int = 0
    qty: float = 0.0
    leverage: float = 1.0
    stop_price: float = 0.0
    take_price: float = 0.0
    strategy: str = ""
    reason: str = ""


class BacktestEngine:
    def __init__(
        self,
        config: TradingConfig,
        signal_tf: str,
        regime_tf: str | None,
        strategy_ids: list[str],
        strategy_params: dict[str, dict[str, Any]],
        data_loader: MarketDataLoader | None = None,
        market_bundles: dict[str, SymbolDataBundle] | None = None,
        progress_cb: Callable[[str], None] | None = None,
        progress_name: str = "backtest",
        progress_updates: int = 20,
    ):
        self.cfg = config
        self.signal_tf = signal_tf
        self.regime_tf = regime_tf
        self.strategy_ids = strategy_ids
        self.strategy_params = strategy_params
        self.data_loader = data_loader if data_loader is not None else MarketDataLoader(config.data)
        self.progress_cb = progress_cb
        self.progress_name = progress_name
        self.progress_updates = max(1, int(progress_updates))

        self.execution = ExecutionSimulator(
            fee_rate=config.costs.fee_rate,
            slippage_bps=config.costs.slippage_bps,
        )

        self.strategies = {
            sid: build_strategy(sid, self.strategy_params.get(sid, {})) for sid in self.strategy_ids
        }

        self.market_bundles = (
            market_bundles
            if market_bundles is not None
            else load_market_bundles(
                config.data,
                signal_tf=signal_tf,
                regime_tf=regime_tf,
                start=config.data.start,
                end=config.data.end,
                warmup_bars=config.data.warmup_bars,
                loader=self.data_loader,
            )
        )

        self.feature_frames: dict[str, dict[str, pd.DataFrame]] = {sid: {} for sid in self.strategy_ids}
        self.feature_indexed: dict[str, dict[str, pd.DataFrame]] = {sid: {} for sid in self.strategy_ids}
        self.exec_frames: dict[str, pd.DataFrame] = {}
        self.exec_indexed: dict[str, pd.DataFrame] = {}
        self.exec_open_times: dict[str, np.ndarray] = {}
        self.exec_close_times: dict[str, np.ndarray] = {}
        self.funding_maps: dict[str, pd.Series] = {}

        self._prepare_frames()

    def _progress(self, message: str):
        if self.progress_cb is not None:
            self.progress_cb(message)

    def _prepare_frames(self):
        start = self.cfg.data.start
        end = self.cfg.data.end

        for symbol, bundle in self.market_bundles.items():
            exec_df = bundle.execution.copy().sort_values("close_time").reset_index(drop=True)
            exec_df = exec_df[(exec_df["close_time"] >= start) & (exec_df["close_time"] <= end)].copy()
            if exec_df.empty:
                continue

            funding_df = bundle.funding.copy().sort_values("close_time").reset_index(drop=True)
            funding_df = funding_df[(funding_df["close_time"] >= start) & (funding_df["close_time"] <= end)].copy()

            self.exec_frames[symbol] = exec_df
            self.exec_indexed[symbol] = exec_df.set_index("close_time")
            self.exec_open_times[symbol] = exec_df["open_time"].values
            self.exec_close_times[symbol] = exec_df["close_time"].values
            self.funding_maps[symbol] = (
                funding_df.set_index("close_time")["fundingRate"] if not funding_df.empty else pd.Series(dtype=float)
            )

            for sid, strategy in self.strategies.items():
                features = build_feature_frame(bundle.signal, bundle.regime, strategy.params)
                features = features[features["close_time"] <= end].copy().sort_values("close_time").reset_index(drop=True)
                self.feature_frames[sid][symbol] = features
                self.feature_indexed[sid][symbol] = features.set_index("close_time")

    def _build_timeline(self, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
        all_ts = []
        for _, frame in self.exec_frames.items():
            part = frame[(frame["close_time"] >= start) & (frame["close_time"] <= end)]
            all_ts.append(part["close_time"])
        if not all_ts:
            return []
        return pd.concat(all_ts).drop_duplicates().sort_values().tolist()

    def _next_exec_close_time(self, symbol: str, signal_time: pd.Timestamp) -> pd.Timestamp | None:
        # timestamp semantics: signal time is signal bar close_time.
        # first fill is earliest execution bar where open_time >= signal_time.
        open_arr = self.exec_open_times[symbol]
        idx = open_arr.searchsorted(np.datetime64(signal_time), side="left")
        if idx >= len(open_arr):
            return None
        return pd.Timestamp(self.exec_close_times[symbol][idx])

    def _aggregate_decisions(self, symbol: str, ts: pd.Timestamp, current_side: int) -> tuple[int, StrategyDecision | None, str]:
        decisions: list[tuple[str, StrategyDecision]] = []
        for sid in self.strategy_ids:
            indexed = self.feature_indexed[sid].get(symbol)
            if indexed is None or ts not in indexed.index:
                continue
            row = indexed.loc[ts]
            dec = self.strategies[sid].decide(row, current_side)
            decisions.append((sid, dec))

        if not decisions:
            return FLAT, None, "no_strategy_data"

        signed = 0.0
        for _, dec in decisions:
            signed += dec.side * max(0.0, float(dec.strength)) * max(0.0, float(dec.confidence))

        if abs(signed) < 0.15:
            best_sid, best_dec = max(decisions, key=lambda x: x[1].confidence)
            return FLAT, best_dec, best_sid

        target_side = LONG if signed > 0 else SHORT
        candidates = [(sid, dec) for sid, dec in decisions if dec.side == target_side]
        if not candidates:
            best_sid, best_dec = max(decisions, key=lambda x: abs(x[1].strength))
            return FLAT, best_dec, best_sid

        best_sid, best_dec = max(candidates, key=lambda x: x[1].strength * x[1].confidence)
        return target_side, best_dec, best_sid

    def _build_open_order(
        self,
        symbol: str,
        target_side: int,
        strategy_id: str,
        decision: StrategyDecision,
        signal_time: pd.Timestamp,
        next_exec_close_time: pd.Timestamp,
        feature_row: pd.Series,
        ledger: PortfolioLedger,
        mark_prices: dict[str, float],
    ) -> ScheduledOrder | None:
        exec_idx = self.exec_indexed[symbol]
        if next_exec_close_time not in exec_idx.index:
            return None

        raw_open = float(exec_idx.loc[next_exec_close_time, "open"])
        est_fill = self.execution.apply_slippage(raw_open, side_to_open_order(target_side))

        atr_value = float(feature_row.get("atr", np.nan))
        if not np.isfinite(atr_value) or atr_value <= 0:
            return None

        stop_dist = max(1e-9, decision.stop_atr * atr_value)
        take_dist = max(1e-9, decision.take_atr * atr_value)

        equity = ledger.mark_to_market(mark_prices)
        drawdown = ledger.current_drawdown(mark_prices)
        risk_scale = drawdown_risk_scale(
            drawdown,
            self.cfg.portfolio.dd_soft,
            self.cfg.portfolio.dd_hard,
            self.cfg.portfolio.min_risk_scale,
        )

        risk_budget = equity * self.cfg.portfolio.risk_per_trade * risk_scale
        risk_per_unit = stop_dist / max(est_fill, 1e-12)
        if risk_per_unit <= 0:
            return None

        notional_risk = risk_budget / risk_per_unit
        gross = ledger.gross_notional(mark_prices)
        symbol_now = ledger.symbol_notional(symbol, mark_prices)

        cap_total = max(0.0, equity * self.cfg.portfolio.max_gross_leverage - gross)
        cap_symbol = max(0.0, equity * self.cfg.portfolio.max_symbol_weight - symbol_now)

        if len(ledger.positions) >= self.cfg.portfolio.max_total_positions:
            return None

        notional = min(notional_risk, cap_total, cap_symbol)
        if notional <= 10.0:
            return None

        qty = notional / est_fill
        leverage = max(1.0, notional / max(equity, 1e-9))

        if target_side == LONG:
            stop_price = est_fill - stop_dist
            take_price = est_fill + take_dist
        else:
            stop_price = est_fill + stop_dist
            take_price = est_fill - take_dist

        if stop_price <= 0 or take_price <= 0:
            return None

        return ScheduledOrder(
            kind="open",
            symbol=symbol,
            execution_close_time=next_exec_close_time,
            signal_time=signal_time,
            side=target_side,
            qty=qty,
            leverage=leverage,
            stop_price=stop_price,
            take_price=take_price,
            strategy=strategy_id,
            reason=decision.reason,
        )

    @staticmethod
    def _enqueue(queue: dict[pd.Timestamp, list[ScheduledOrder]], order: ScheduledOrder):
        queue.setdefault(order.execution_close_time, []).append(order)

    def _execute_due_orders(
        self,
        ts: pd.Timestamp,
        queue: dict[pd.Timestamp, list[ScheduledOrder]],
        ledger: PortfolioLedger,
        mark_prices: dict[str, float],
    ):
        due = queue.pop(ts, [])
        if not due:
            return

        due.sort(key=lambda x: 0 if x.kind == "close" else 1)

        for order in due:
            exec_df = self.exec_indexed.get(order.symbol)
            if exec_df is None or ts not in exec_df.index:
                continue

            bar = exec_df.loc[ts]
            raw_open = float(bar["open"])

            if order.kind == "close":
                pos = ledger.positions.get(order.symbol)
                if pos is None:
                    continue
                fill = self.execution.apply_slippage(raw_open, side_to_close_order(pos.side))
                ledger.close_position(order.symbol, ts, fill, raw_open, order.reason)
                continue

            if order.symbol in ledger.positions:
                continue

            equity = ledger.mark_to_market(mark_prices)
            gross = ledger.gross_notional(mark_prices)
            symbol_notional = ledger.symbol_notional(order.symbol, mark_prices)
            raw_notional = abs(raw_open * order.qty)

            if len(ledger.positions) >= self.cfg.portfolio.max_total_positions:
                continue
            if gross + raw_notional > equity * self.cfg.portfolio.max_gross_leverage:
                continue
            if symbol_notional + raw_notional > equity * self.cfg.portfolio.max_symbol_weight:
                continue

            fill = self.execution.apply_slippage(raw_open, side_to_open_order(order.side))
            ledger.open_position(
                symbol=order.symbol,
                side=order.side,
                timestamp=ts,
                signal_time=order.signal_time,
                strategy=order.strategy,
                entry_reason=order.reason,
                entry_price=fill,
                raw_price=raw_open,
                qty=order.qty,
                leverage=order.leverage,
                stop_price=order.stop_price,
                take_price=order.take_price,
            )

    def _apply_funding(self, ts: pd.Timestamp, ledger: PortfolioLedger, mark_prices: dict[str, float]):
        if not self.cfg.costs.funding_enabled:
            return
        for symbol, pos in list(ledger.positions.items()):
            funding = self.funding_maps.get(symbol)
            if funding is None or ts not in funding.index:
                continue
            mark = float(mark_prices.get(symbol, pos.entry_price))
            ledger.apply_funding(symbol, float(funding.loc[ts]), mark)

    def _check_stop_take(self, ts: pd.Timestamp, ledger: PortfolioLedger):
        for symbol in list(ledger.positions.keys()):
            pos = ledger.positions.get(symbol)
            if pos is None:
                continue
            exec_df = self.exec_indexed.get(symbol)
            if exec_df is None or ts not in exec_df.index:
                continue

            bar = exec_df.loc[ts]
            high = float(bar["high"])
            low = float(bar["low"])

            hit_price = None
            reason = None

            if pos.side == LONG:
                stop_hit = low <= pos.stop_price
                take_hit = high >= pos.take_price
                if stop_hit:
                    hit_price = pos.stop_price
                    reason = "stop_loss"
                elif take_hit:
                    hit_price = pos.take_price
                    reason = "take_profit"
            else:
                stop_hit = high >= pos.stop_price
                take_hit = low <= pos.take_price
                if stop_hit:
                    hit_price = pos.stop_price
                    reason = "stop_loss"
                elif take_hit:
                    hit_price = pos.take_price
                    reason = "take_profit"

            if hit_price is None:
                continue

            fill = self.execution.apply_slippage(float(hit_price), side_to_close_order(pos.side))
            ledger.close_position(symbol, ts, fill, float(hit_price), reason)

    def _max_hold_limit(self) -> int:
        vals = []
        for sid in self.strategy_ids:
            vals.append(int(self.strategies[sid].params.get("max_hold", 120)))
        return min(vals) if vals else 120

    def _generate_signals(
        self,
        ts: pd.Timestamp,
        queue: dict[pd.Timestamp, list[ScheduledOrder]],
        ledger: PortfolioLedger,
        mark_prices: dict[str, float],
    ):
        if ts < self.cfg.data.start:
            return

        max_hold = self._max_hold_limit()

        for symbol in self.cfg.data.universe:
            if symbol not in self.exec_indexed:
                continue

            has_signal_point = any(ts in self.feature_indexed[sid].get(symbol, pd.DataFrame()).index for sid in self.strategy_ids)
            if not has_signal_point:
                continue

            if symbol in ledger.positions:
                ledger.increment_holding_bar(symbol)

            position = ledger.positions.get(symbol)
            current_side = position.side if position is not None else FLAT

            target_side, chosen_decision, source_sid = self._aggregate_decisions(symbol, ts, current_side)
            if chosen_decision is None:
                continue

            if position is not None and position.holding_bars >= max_hold:
                target_side = FLAT
                chosen_decision = StrategyDecision(
                    side=FLAT,
                    strength=0.0,
                    stop_atr=chosen_decision.stop_atr,
                    take_atr=chosen_decision.take_atr,
                    confidence=1.0,
                    reason="max_hold_exit",
                )

            next_exec = self._next_exec_close_time(symbol, ts)
            if next_exec is None:
                continue

            feature_row = self.feature_indexed[source_sid][symbol].loc[ts]

            if position is None:
                if target_side == FLAT:
                    continue
                open_order = self._build_open_order(
                    symbol=symbol,
                    target_side=target_side,
                    strategy_id=source_sid,
                    decision=chosen_decision,
                    signal_time=ts,
                    next_exec_close_time=next_exec,
                    feature_row=feature_row,
                    ledger=ledger,
                    mark_prices=mark_prices,
                )
                if open_order is not None:
                    self._enqueue(queue, open_order)
                continue

            if target_side == position.side:
                continue

            self._enqueue(
                queue,
                ScheduledOrder(
                    kind="close",
                    symbol=symbol,
                    execution_close_time=next_exec,
                    signal_time=ts,
                    strategy=position.strategy,
                    reason=chosen_decision.reason,
                ),
            )

            if target_side == FLAT:
                continue

            open_order = self._build_open_order(
                symbol=symbol,
                target_side=target_side,
                strategy_id=source_sid,
                decision=chosen_decision,
                signal_time=ts,
                next_exec_close_time=next_exec,
                feature_row=feature_row,
                ledger=ledger,
                mark_prices=mark_prices,
            )
            if open_order is not None:
                open_order.reason = f"flip_{open_order.reason}"
                self._enqueue(queue, open_order)

    def run(self) -> BacktestResult:
        start = self.cfg.data.start
        end = self.cfg.data.end

        ledger = PortfolioLedger(self.cfg.portfolio.initial_balance, self.execution)
        timeline = self._build_timeline(start, end)
        if not timeline:
            raise ValueError("No execution data in selected time range")

        total_steps = len(timeline)
        step_interval = max(1, total_steps // self.progress_updates)
        self._progress(
            f"{self.progress_name}: timeline ready | steps={total_steps} "
            f"symbols={len(self.exec_indexed)} period={start} -> {end}"
        )

        queue: dict[pd.Timestamp, list[ScheduledOrder]] = {}
        mark_prices: dict[str, float] = {}

        for idx, ts in enumerate(timeline, start=1):
            for symbol, exec_idx in self.exec_indexed.items():
                if ts in exec_idx.index:
                    mark_prices[symbol] = float(exec_idx.loc[ts, "close"])

            self._execute_due_orders(ts, queue, ledger, mark_prices)
            self._apply_funding(ts, ledger, mark_prices)
            self._check_stop_take(ts, ledger)
            self._generate_signals(ts, queue, ledger, mark_prices)
            ledger.snapshot(ts, mark_prices)

            if idx == 1 or idx == total_steps or idx % step_interval == 0:
                pct = 100.0 * idx / max(total_steps, 1)
                self._progress(
                    f"{self.progress_name}: {idx}/{total_steps} ({pct:.1f}%) "
                    f"ts={ts} open_positions={len(ledger.positions)}"
                )

        final_ts = timeline[-1]
        for symbol in list(ledger.positions.keys()):
            exec_idx = self.exec_indexed[symbol]
            if final_ts not in exec_idx.index:
                continue
            raw_close = float(exec_idx.loc[final_ts, "close"])
            pos = ledger.positions[symbol]
            fill = self.execution.apply_slippage(raw_close, side_to_close_order(pos.side))
            ledger.close_position(symbol, final_ts, fill, raw_close, "final_close")

        ledger.snapshot(final_ts, mark_prices)

        equity_df = ledger.equity_frame()
        trades_df = ledger.trades_frame()
        metrics = compute_metrics(equity_df, trades_df, self.cfg.data.exec_tf)
        self._progress(
            f"{self.progress_name}: completed | sharpe={metrics.sharpe:.3f} "
            f"max_dd={metrics.max_drawdown*100:.2f}% trades={metrics.trade_count}"
        )

        symbol_exec = {
            symbol: df[(df["close_time"] >= start) & (df["close_time"] <= end)].copy()
            for symbol, df in self.exec_frames.items()
        }

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_df,
            trades=trades_df,
            strategy_bundle={
                "signal_tf": self.signal_tf,
                "regime_tf": self.regime_tf,
                "strategy_ids": list(self.strategy_ids),
                "strategy_params": self.strategy_params,
            },
            symbol_execution=symbol_exec,
        )
