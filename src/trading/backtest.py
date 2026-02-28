from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from .config import TradingConfig
from .data_loader import load_market_store, MarketStore
from .execution import ExecutionModel, side_to_open_order_side, side_to_close_order_side
from .metrics import compute_metrics
from .portfolio import Portfolio
from .signals import StrategyParams, HybridTrendMeanRevStrategy
from .types import BacktestResult, FLAT, LONG, SHORT


@dataclass
class ScheduledOrder:
    """待执行订单对象（由信号产生，在未来时间点执行）。"""
    kind: str  # open / close
    symbol: str
    timestamp: pd.Timestamp
    signal_time: pd.Timestamp
    side: int = 0
    qty: float = 0.0
    leverage: float = 1.0
    stop_price: float = 0.0
    take_price: float = 0.0
    reason: str = ""


class Backtester:
    """回测主引擎：串联信号、撮合、风控、记账与绩效统计。"""
    def __init__(
        self,
        config: TradingConfig,
        strategy_params: Dict[str, Any] | None = None,
        market_store: MarketStore | None = None,
    ):
        """初始化回测器并预处理数据与索引。"""
        self.config = config
        self.market_store = market_store if market_store is not None else load_market_store(config.data)

        raw_params = strategy_params if strategy_params is not None else (config.strategy.params or {})
        params = StrategyParams.from_dict(raw_params)
        if "stop_atr_mult" not in raw_params:
            params.stop_atr_mult = config.risk.stop_atr_mult
        if "take_atr_mult" not in raw_params:
            params.take_atr_mult = config.risk.take_atr_mult
        self.strategy_params = params
        self.strategy = HybridTrendMeanRevStrategy(params)

        self.exec_model = ExecutionModel(
            fee_rate=config.costs.fee_rate,
            slippage_bps=config.costs.slippage_bps,
        )

        self.signal_frames: Dict[str, pd.DataFrame] = {}
        self.exec_frames: Dict[str, pd.DataFrame] = {}
        self.funding_maps: Dict[str, pd.Series] = {}
        self.exec_indexed: Dict[str, pd.DataFrame] = {}
        self.signal_indexed: Dict[str, pd.DataFrame] = {}
        self.exec_timestamps: Dict[str, np.ndarray] = {}

        self._prepare_frames()

    def _prepare_frames(self):
        """预计算信号数据、执行数据、资金费序列及时间索引缓存。"""
        start_time = self.config.data.start_time
        end_time = self.config.data.end_time

        for symbol, frames in self.market_store.items():
            signal_df = self.strategy.prepare(frames.signal)
            signal_df = signal_df[(signal_df["timestamp"] <= end_time)].copy()
            signal_df = signal_df.sort_values("timestamp").reset_index(drop=True)

            exec_df = frames.execution.copy().sort_values("timestamp").reset_index(drop=True)
            exec_df = exec_df[(exec_df["timestamp"] >= start_time) & (exec_df["timestamp"] <= end_time)].copy()

            funding_df = frames.funding.copy().sort_values("timestamp").reset_index(drop=True)
            funding_df = funding_df[(funding_df["timestamp"] >= start_time) & (funding_df["timestamp"] <= end_time)].copy()

            self.signal_frames[symbol] = signal_df
            self.exec_frames[symbol] = exec_df
            self.funding_maps[symbol] = funding_df.set_index("timestamp")["fundingRate"] if not funding_df.empty else pd.Series(dtype=float)
            self.exec_indexed[symbol] = exec_df.set_index("timestamp")
            self.signal_indexed[symbol] = signal_df.set_index("timestamp")
            self.exec_timestamps[symbol] = exec_df["timestamp"].values

    def _build_timeline(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[pd.Timestamp]:
        """构建统一执行时间轴（各交易对执行周期时间并集）。"""
        all_ts = []
        for symbol, df in self.exec_frames.items():
            subset = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]
            all_ts.append(subset["timestamp"])
        if not all_ts:
            return []
        timeline = pd.concat(all_ts).drop_duplicates().sort_values().tolist()
        return timeline

    def _next_exec_timestamp(self, symbol: str, ts: pd.Timestamp) -> pd.Timestamp | None:
        """返回某个信号时刻之后的下一根执行 K 线时间。"""
        arr = self.exec_timestamps[symbol]
        idx = arr.searchsorted(np.datetime64(ts), side="right")
        if idx >= len(arr):
            return None
        return pd.Timestamp(arr[idx])

    def _build_open_order(
        self,
        symbol: str,
        side: int,
        signal_time: pd.Timestamp,
        next_exec_time: pd.Timestamp,
        row: pd.Series,
        equity: float,
        gross_notional: float,
        open_positions_count: int,
    ) -> ScheduledOrder | None:
        """根据风控约束和 ATR 止损距离构造开仓订单。"""
        # 1) 获取下一根执行 K 线开盘价，用于估算成交价与仓位规模。
        exec_df = self.exec_indexed[symbol]
        if next_exec_time not in exec_df.index:
            return None

        raw_open = float(exec_df.loc[next_exec_time, "open"])
        order_side = side_to_open_order_side(side)
        est_fill = self.exec_model.apply_slippage(raw_open, order_side)

        # 2) 读取 ATR 并计算止损/止盈距离；ATR 无效时放弃开仓。
        atr = float(row["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return None

        stop_dist = self.strategy_params.stop_atr_mult * atr
        take_dist = self.strategy_params.take_atr_mult * atr
        if stop_dist <= 0 or take_dist <= 0:
            return None

        # 3) 风险预算转仓位：先按每单风险上限算可开名义价值。
        risk_budget = equity * self.config.risk.risk_per_trade
        risk_per_unit = stop_dist / max(est_fill, 1e-12)
        if risk_per_unit <= 0:
            return None
        notional_risk = risk_budget / risk_per_unit

        # 4) 叠加组合层约束：总杠杆、最大持仓数、最小有效名义金额。
        max_total_notional = equity * self.config.risk.max_gross_leverage
        available_notional = max(0.0, max_total_notional - gross_notional)

        if open_positions_count >= self.config.risk.max_positions:
            return None

        notional = min(notional_risk, available_notional)
        if notional <= 10.0:
            return None

        # 5) 确定数量、杠杆、止损止盈价格，并做价格有效性检查。
        qty = notional / est_fill
        leverage = max(1, int(np.ceil(notional / max(equity, 1e-9))))

        if side == LONG:
            stop_price = est_fill - stop_dist
            take_price = est_fill + take_dist
        else:
            stop_price = est_fill + stop_dist
            take_price = est_fill - take_dist

        if stop_price <= 0 or take_price <= 0:
            return None

        # 6) 返回待执行开仓订单（实际成交在未来执行环节发生）。
        return ScheduledOrder(
            kind="open",
            symbol=symbol,
            timestamp=next_exec_time,
            signal_time=signal_time,
            side=side,
            qty=qty,
            leverage=leverage,
            stop_price=stop_price,
            take_price=take_price,
            reason="signal_entry",
        )

    def _schedule(self, queue: Dict[pd.Timestamp, list[ScheduledOrder]], order: ScheduledOrder):
        """将订单加入对应时间点的待执行队列。"""
        queue.setdefault(order.timestamp, []).append(order)

    def _execute_due_orders(
        self,
        ts: pd.Timestamp,
        queue: Dict[pd.Timestamp, list[ScheduledOrder]],
        portfolio: Portfolio,
        mark_prices: Dict[str, float],
    ):
        """执行当前时刻到期订单，优先处理平仓再处理开仓。"""
        # 1) 取出当前时刻待执行订单；空队列直接返回。
        due = queue.pop(ts, [])
        if not due:
            return

        # 2) 同时刻冲突处理：先平仓后开仓，避免旧仓位占用风险额度。
        due.sort(key=lambda x: 0 if x.kind == "close" else 1)

        for order in due:
            exec_df = self.exec_indexed[order.symbol]
            if ts not in exec_df.index:
                continue

            raw_open = float(exec_df.loc[ts, "open"])

            # 3) 平仓订单：按当前开盘价 + 滑点成交，并立即结算。
            if order.kind == "close":
                pos = portfolio.positions.get(order.symbol)
                if pos is None:
                    continue
                side = side_to_close_order_side(pos.side)
                fill = self.exec_model.apply_slippage(raw_open, side)
                portfolio.close_position(order.symbol, order.signal_time, fill, order.reason, raw_open)
                continue

            # 4) 开仓订单：再次检查仓位存在性与组合风险约束（双重保护）。
            if order.symbol in portfolio.positions:
                continue

            equity = portfolio.mark_to_market(mark_prices)
            gross = portfolio.gross_notional(mark_prices)
            open_count = len(portfolio.positions)
            new_notional = order.qty * raw_open
            if open_count >= self.config.risk.max_positions:
                continue
            if gross + new_notional > equity * self.config.risk.max_gross_leverage:
                continue

            # 5) 通过约束后执行开仓，并把成交写入组合账本。
            order_side = side_to_open_order_side(order.side)
            fill = self.exec_model.apply_slippage(raw_open, order_side)
            portfolio.open_position(
                symbol=order.symbol,
                side=order.side,
                timestamp=ts,
                signal_time=order.signal_time,
                entry_price=fill,
                qty=order.qty,
                leverage=order.leverage,
                stop_price=order.stop_price,
                take_price=order.take_price,
                reason=order.reason,
                raw_price=raw_open,
            )

    def _apply_funding(self, ts: pd.Timestamp, portfolio: Portfolio, mark_prices: Dict[str, float]):
        """在当前时刻对持仓应用资金费结算。"""
        if not self.config.costs.funding_rate_enabled:
            return
        for symbol, pos in list(portfolio.positions.items()):
            series = self.funding_maps[symbol]
            if ts not in series.index:
                continue
            mark_px = mark_prices.get(symbol, pos.entry_price)
            portfolio.apply_funding(symbol, float(series.loc[ts]), float(mark_px))

    def _check_stop_take(self, ts: pd.Timestamp, portfolio: Portfolio):
        """检查止损/止盈是否触发；同 bar 触发时优先止损（保守）。"""
        for symbol in list(portfolio.positions.keys()):
            pos = portfolio.positions.get(symbol)
            if pos is None:
                continue
            exec_df = self.exec_indexed[symbol]
            if ts not in exec_df.index:
                continue

            bar = exec_df.loc[ts]
            high = float(bar["high"])
            low = float(bar["low"])

            hit_price = None
            reason = None
            # 对同一根 bar 内同时触发止损与止盈的情况，按更保守假设优先止损。
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

            # 触发后按触发价加滑点进行平仓，并立即落账。
            side = side_to_close_order_side(pos.side)
            fill = self.exec_model.apply_slippage(float(hit_price), side)
            portfolio.close_position(symbol, ts, fill, reason, float(hit_price))

    def _generate_signals(
        self,
        ts: pd.Timestamp,
        queue: Dict[pd.Timestamp, list[ScheduledOrder]],
        portfolio: Portfolio,
        mark_prices: Dict[str, float],
    ):
        """在信号周期时间点生成决策并调度未来执行订单。"""
        # 1) 起始时间保护：回测起点之前不允许下单。
        if ts < self.config.data.start_time:
            return

        for symbol, signal_df in self.signal_indexed.items():
            if ts not in signal_df.index:
                continue

            # 2) 若已持仓，先更新持仓 bar 计数，供超时出场规则使用。
            if symbol in portfolio.positions:
                portfolio.increment_holding_bar(symbol)

            # 3) 生成当前信号决策，并定位下一执行时间。
            row = signal_df.loc[ts]
            position = portfolio.positions.get(symbol)
            decision = self.strategy.decide(row, position)
            next_exec_time = self._next_exec_timestamp(symbol, ts)
            if next_exec_time is None:
                continue

            # 4) 无持仓场景：仅在出现开仓信号时调度开仓订单。
            if position is None:
                if decision.target_side == FLAT:
                    continue
                equity = portfolio.mark_to_market(mark_prices)
                gross = portfolio.gross_notional(mark_prices)
                open_order = self._build_open_order(
                    symbol=symbol,
                    side=decision.target_side,
                    signal_time=ts,
                    next_exec_time=next_exec_time,
                    row=row,
                    equity=equity,
                    gross_notional=gross,
                    open_positions_count=len(portfolio.positions),
                )
                if open_order is not None:
                    open_order.reason = decision.reason
                    self._schedule(queue, open_order)
                continue

            # 5) 有持仓场景：方向未变则继续持有，无需调度。
            if decision.target_side == position.side:
                continue

            # 6) 方向变化或平仓信号：先调度平仓。
            self._schedule(
                queue,
                ScheduledOrder(
                    kind="close",
                    symbol=symbol,
                    timestamp=next_exec_time,
                    signal_time=ts,
                    reason=decision.reason,
                ),
            )

            if decision.target_side == FLAT:
                continue

            # 7) 若信号要求反手，再额外调度一笔开仓订单（flip）。
            equity = portfolio.mark_to_market(mark_prices)
            gross = portfolio.gross_notional(mark_prices)
            open_order = self._build_open_order(
                symbol=symbol,
                side=decision.target_side,
                signal_time=ts,
                next_exec_time=next_exec_time,
                row=row,
                equity=equity,
                gross_notional=gross,
                open_positions_count=max(0, len(portfolio.positions) - 1),
            )
            if open_order is not None:
                open_order.reason = f"flip_{decision.reason}"
                self._schedule(queue, open_order)

    def run(self, start_time: pd.Timestamp | None = None, end_time: pd.Timestamp | None = None) -> BacktestResult:
        """运行单次回测并返回包含指标、曲线和交易记录的结果对象。"""
        # 1) 解析运行区间并初始化组合状态。
        start = self.config.data.start_time if start_time is None else pd.Timestamp(start_time)
        end = self.config.data.end_time if end_time is None else pd.Timestamp(end_time)

        portfolio = Portfolio(self.config.risk.initial_balance, self.exec_model)
        timeline = self._build_timeline(start, end)
        if not timeline:
            raise ValueError("回测时间段内没有可用执行数据")

        # 2) 准备运行期状态：订单队列与最新标记价格缓存。
        queue: Dict[pd.Timestamp, list[ScheduledOrder]] = {}
        mark_prices: Dict[str, float] = {}

        # 3) 主循环：每个执行时刻按固定顺序驱动回测状态机。
        # 顺序设计为：更新价格 -> 执行到期订单 -> 资金费 -> 止盈止损 -> 生成新信号 -> 记快照。
        for ts in timeline:
            for symbol, exec_df in self.exec_indexed.items():
                if ts in exec_df.index:
                    mark_prices[symbol] = float(exec_df.loc[ts, "close"])

            self._execute_due_orders(ts, queue, portfolio, mark_prices)
            self._apply_funding(ts, portfolio, mark_prices)
            self._check_stop_take(ts, portfolio)
            self._generate_signals(ts, queue, portfolio, mark_prices)

            portfolio.snapshot(ts, mark_prices)

        # 4) 回测结束清仓：将剩余持仓按最后时刻价格强制平仓。
        final_ts = timeline[-1]
        for symbol in list(portfolio.positions.keys()):
            exec_df = self.exec_indexed[symbol]
            if final_ts not in exec_df.index:
                continue
            raw_close = float(exec_df.loc[final_ts, "close"])
            pos = portfolio.positions[symbol]
            side = side_to_close_order_side(pos.side)
            fill = self.exec_model.apply_slippage(raw_close, side)
            portfolio.close_position(symbol, final_ts, fill, "final_close", raw_close)

        # 5) 收尾快照 + 结果组装：构建权益、交易明细与绩效指标。
        portfolio.snapshot(final_ts, mark_prices)

        equity_df = portfolio.equity_frame()
        trades_df = portfolio.trades_frame()
        metrics = compute_metrics(equity_df, trades_df, self.config.data.timeframe_execution)

        symbol_exec = {
            symbol: df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
            for symbol, df in self.exec_frames.items()
        }

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_df,
            trades=trades_df,
            strategy_params=self.strategy_params.to_dict(),
            symbol_execution=symbol_exec,
        )
