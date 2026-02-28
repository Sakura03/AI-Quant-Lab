from __future__ import annotations

from typing import Any, Callable

import copy
import json

import numpy as np
import pandas as pd

from trading.backtest.engine import BacktestEngine
from trading.config import TradingConfig
from trading.data.loader import MarketDataLoader
from trading.domain.types import ExperimentResult, WindowResult
from trading.evaluate.metrics import compute_metrics, score_metrics
from trading.optimize.walkforward import WalkForwardWindow, make_walk_forward_windows
from trading.strategies.factory import mutate_strategy_params, sample_strategy_params


class WalkForwardEvolver:
    def __init__(self, config: TradingConfig, progress_cb: Callable[[str], None] | None = None):
        self.cfg = config
        self.rng = np.random.default_rng(config.engine.seed)
        self.loader = MarketDataLoader(config.data)
        self.progress_cb = progress_cb

    def _progress(self, message: str):
        if self.progress_cb is not None:
            self.progress_cb(message)

    def _clone_cfg_with_period(self, start: pd.Timestamp, end: pd.Timestamp) -> TradingConfig:
        raw = copy.deepcopy(self.cfg.to_dict())
        raw["data"]["start"] = pd.Timestamp(start).strftime("%Y%m%d-%H%M%S")
        raw["data"]["end"] = pd.Timestamp(end).strftime("%Y%m%d-%H%M%S")
        return TradingConfig.from_dict(raw)

    def _sample_strategy_subset(self) -> list[str]:
        pool = list(self.cfg.strategies.enabled)
        k = int(self.rng.integers(1, len(pool) + 1))
        picked = [str(x) for x in self.rng.choice(pool, size=k, replace=False).tolist()]
        return sorted(picked)

    def _sample_genome(self) -> dict[str, Any]:
        strategy_ids = self._sample_strategy_subset()
        signal_tf = str(self.rng.choice(self.cfg.data.signal_tfs))
        regime_candidates = list(self.cfg.data.regime_tfs) if self.cfg.data.regime_tfs else [None]
        regime_tf = str(self.rng.choice(regime_candidates)) if regime_candidates else None
        params = {sid: sample_strategy_params(sid, self.rng) for sid in strategy_ids}
        return {
            "strategy_ids": strategy_ids,
            "signal_tf": signal_tf,
            "regime_tf": regime_tf,
            "strategy_params": params,
        }

    def _mutate_genome(self, genome: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(genome)
        m = self.cfg.optimizer.mutation_strength

        if self.rng.random() < 0.20 * m:
            out["signal_tf"] = str(self.rng.choice(self.cfg.data.signal_tfs))

        if self.cfg.data.regime_tfs and self.rng.random() < 0.20 * m:
            out["regime_tf"] = str(self.rng.choice(self.cfg.data.regime_tfs))

        if self.rng.random() < 0.15 * m and len(self.cfg.strategies.enabled) > 1:
            out["strategy_ids"] = self._sample_strategy_subset()

        for sid in list(out["strategy_ids"]):
            base = out["strategy_params"].get(sid)
            if base is None:
                out["strategy_params"][sid] = sample_strategy_params(sid, self.rng)
            else:
                out["strategy_params"][sid] = mutate_strategy_params(sid, base, m, self.rng)

        out["strategy_params"] = {sid: out["strategy_params"][sid] for sid in out["strategy_ids"]}
        return out

    def _crossover(self, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        child = {}
        child["signal_tf"] = a["signal_tf"] if self.rng.random() < 0.5 else b["signal_tf"]
        child["regime_tf"] = a["regime_tf"] if self.rng.random() < 0.5 else b["regime_tf"]

        union_ids = sorted(set(a["strategy_ids"]) | set(b["strategy_ids"]))
        picked = []
        for sid in union_ids:
            if self.rng.random() < 0.5:
                picked.append(sid)
        if not picked:
            picked = [str(self.rng.choice(union_ids))]
        child["strategy_ids"] = sorted(picked)

        params = {}
        for sid in child["strategy_ids"]:
            if sid in a["strategy_params"] and sid in b["strategy_params"]:
                base = a["strategy_params"][sid] if self.rng.random() < 0.5 else b["strategy_params"][sid]
            elif sid in a["strategy_params"]:
                base = a["strategy_params"][sid]
            elif sid in b["strategy_params"]:
                base = b["strategy_params"][sid]
            else:
                base = sample_strategy_params(sid, self.rng)
            params[sid] = copy.deepcopy(base)
        child["strategy_params"] = params
        return self._mutate_genome(child)

    def _run_period_backtest(self, genome: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp):
        period_cfg = self._clone_cfg_with_period(start, end)
        engine = BacktestEngine(
            config=period_cfg,
            signal_tf=genome["signal_tf"],
            regime_tf=genome["regime_tf"],
            strategy_ids=genome["strategy_ids"],
            strategy_params=genome["strategy_params"],
            data_loader=self.loader,
            progress_cb=None,
        )
        return engine.run()

    def _eval_genome(self, genome: dict[str, Any], win: WalkForwardWindow) -> dict[str, Any]:
        train_res = self._run_period_backtest(genome, win.train_start, win.train_end)
        val_res = self._run_period_backtest(genome, win.val_start, win.val_end)

        max_symbol_share = 0.0
        if not val_res.equity_curve.empty and "max_symbol_share" in val_res.equity_curve.columns:
            max_symbol_share = float(val_res.equity_curve["max_symbol_share"].max())

        score = score_metrics(
            train_metrics=train_res.metrics,
            eval_metrics=val_res.metrics,
            max_symbol_share=max_symbol_share,
            objective_weights=self.cfg.objective.weights,
            hard_limits=self.cfg.objective.hard_limits,
            min_trades=self.cfg.optimizer.min_trades_per_window,
        )

        return {
            "genome": genome,
            "train": train_res,
            "val": val_res,
            "score": score,
        }

    @staticmethod
    def _serialize_genome(genome: dict[str, Any]) -> str:
        return json.dumps(genome, ensure_ascii=False, sort_keys=True)

    def _stitch_oos(self, window_test_results: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        stitched_parts = []
        stitched_trades = []
        capital = self.cfg.portfolio.initial_balance

        for row in window_test_results:
            eq = row["test"].equity_curve.copy()
            if eq.empty:
                continue
            eq = eq.sort_values("timestamp").reset_index(drop=True)
            rets = eq["equity"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            seg = eq[["timestamp"]].copy()
            seg["equity"] = capital * (1.0 + rets).cumprod()
            seg["cash"] = seg["equity"]
            seg["open_positions"] = eq["open_positions"].values
            seg["gross_notional"] = eq["gross_notional"].values
            if stitched_parts:
                seg = seg.iloc[1:].copy()
            if not seg.empty:
                capital = float(seg["equity"].iloc[-1])
                stitched_parts.append(seg)

            tr = row["test"].trades.copy()
            if not tr.empty:
                stitched_trades.append(tr)

        stitched_equity = pd.concat(stitched_parts, ignore_index=True) if stitched_parts else pd.DataFrame(
            columns=["timestamp", "equity", "cash", "open_positions", "gross_notional"]
        )
        if not stitched_equity.empty:
            stitched_equity["turnover"] = stitched_equity["gross_notional"].cumsum()
            stitched_equity["drawdown"] = 1.0 - stitched_equity["equity"] / stitched_equity["equity"].cummax()
            stitched_equity["max_symbol_share"] = 0.0

        stitched_trade_df = pd.concat(stitched_trades, ignore_index=True) if stitched_trades else pd.DataFrame()
        return stitched_equity, stitched_trade_df

    def run(self) -> ExperimentResult:
        windows = make_walk_forward_windows(
            start=self.cfg.data.start,
            end=self.cfg.data.end,
            train_months=self.cfg.split.train_months,
            val_months=self.cfg.split.val_months,
            test_months=self.cfg.split.test_months,
            step_months=self.cfg.split.step_months,
            embargo_days=self.cfg.split.embargo_days,
        )
        if not windows:
            raise ValueError("No walk-forward windows available for the selected date range")

        self._progress(
            f"optimize: prepared {len(windows)} windows "
            f"(train={self.cfg.split.train_months}m val={self.cfg.split.val_months}m test={self.cfg.split.test_months}m)"
        )

        all_trials_rows = []
        window_rows = []
        elite_memory: list[dict[str, Any]] = []

        total_windows = len(windows)
        for wid, win in enumerate(windows, start=1):
            self._progress(
                f"optimize: window {wid}/{total_windows} "
                f"train={win.train_start.date()}->{win.train_end.date()} "
                f"val={win.val_start.date()}->{win.val_end.date()} "
                f"test={win.test_start.date()}->{win.test_end.date()}"
            )

            trial_rows: list[dict[str, Any]] = []
            evaluated: list[dict[str, Any]] = []
            best_val_score = -1e18
            report_every = max(1, self.cfg.optimizer.trials_per_window // 10)

            for _ in range(self.cfg.optimizer.population):
                if elite_memory and self.rng.random() < self.cfg.optimizer.prior_adoption_prob:
                    seed = copy.deepcopy(elite_memory[int(self.rng.integers(0, len(elite_memory)))])
                    genome = self._mutate_genome(seed)
                else:
                    genome = self._sample_genome()
                res = self._eval_genome(genome, win)
                evaluated.append(res)
                best_val_score = max(best_val_score, res["score"])
                c = len(evaluated)
                if c == self.cfg.optimizer.population or c == self.cfg.optimizer.trials_per_window or c % report_every == 0:
                    self._progress(
                        f"optimize: window {wid}/{total_windows} trials {c}/{self.cfg.optimizer.trials_per_window} "
                        f"best_val_score={best_val_score:.4f}"
                    )

            while len(evaluated) < self.cfg.optimizer.trials_per_window:
                evaluated.sort(key=lambda x: x["score"], reverse=True)
                elites = evaluated[: self.cfg.optimizer.elite_top_k]
                pa = elites[int(self.rng.integers(0, len(elites)))]
                pb = elites[int(self.rng.integers(0, len(elites)))]
                if self.rng.random() < self.cfg.optimizer.crossover_prob:
                    genome = self._crossover(pa["genome"], pb["genome"])
                else:
                    genome = self._mutate_genome(pa["genome"])
                res = self._eval_genome(genome, win)
                evaluated.append(res)
                best_val_score = max(best_val_score, res["score"])
                c = len(evaluated)
                if c == self.cfg.optimizer.trials_per_window or c % report_every == 0:
                    self._progress(
                        f"optimize: window {wid}/{total_windows} trials {c}/{self.cfg.optimizer.trials_per_window} "
                        f"best_val_score={best_val_score:.4f}"
                    )

            evaluated.sort(key=lambda x: x["score"], reverse=True)
            best = evaluated[0]

            test_res = self._run_period_backtest(best["genome"], win.test_start, win.test_end)
            max_symbol_share = (
                float(test_res.equity_curve["max_symbol_share"].max())
                if not test_res.equity_curve.empty and "max_symbol_share" in test_res.equity_curve.columns
                else 0.0
            )
            test_score = score_metrics(
                train_metrics=best["train"].metrics,
                eval_metrics=test_res.metrics,
                max_symbol_share=max_symbol_share,
                objective_weights=self.cfg.objective.weights,
                hard_limits=self.cfg.objective.hard_limits,
                min_trades=self.cfg.optimizer.min_trades_per_window,
            )
            self._progress(
                f"optimize: window {wid}/{total_windows} done | "
                f"val_score={best['score']:.4f} test_score={test_score:.4f} "
                f"test_sharpe={test_res.metrics.sharpe:.3f} test_mdd={test_res.metrics.max_drawdown*100:.2f}%"
            )

            window_rows.append(
                {
                    "window_id": win.window_id,
                    "genome": copy.deepcopy(best["genome"]),
                    "train": best["train"],
                    "val": best["val"],
                    "test": test_res,
                    "train_score": best["score"],
                    "val_score": best["score"],
                    "test_score": test_score,
                    "window": win,
                }
            )

            elite_memory = [copy.deepcopy(x["genome"]) for x in evaluated[: self.cfg.optimizer.elite_top_k]]

            for tid, trial in enumerate(evaluated, start=1):
                row = {
                    "window_id": win.window_id,
                    "trial_id": tid,
                    "score": trial["score"],
                    "train_sharpe": trial["train"].metrics.sharpe,
                    "train_mdd": trial["train"].metrics.max_drawdown,
                    "val_sharpe": trial["val"].metrics.sharpe,
                    "val_mdd": trial["val"].metrics.max_drawdown,
                    "val_trades": trial["val"].metrics.trade_count,
                    "signal_tf": trial["genome"]["signal_tf"],
                    "regime_tf": trial["genome"]["regime_tf"],
                    "strategy_ids": ",".join(trial["genome"]["strategy_ids"]),
                    "genome_json": self._serialize_genome(trial["genome"]),
                }
                trial_rows.append(row)

            all_trials_rows.extend(trial_rows)

        stitched_eq, stitched_trades = self._stitch_oos(window_rows)
        stitched_metrics = compute_metrics(stitched_eq, stitched_trades, self.cfg.data.exec_tf).to_dict()
        self._progress(
            f"optimize: stitched OOS done | sharpe={stitched_metrics.get('sharpe', 0.0):.3f} "
            f"max_dd={stitched_metrics.get('max_drawdown', 0.0)*100:.2f}% "
            f"trades={int(stitched_metrics.get('trade_count', 0))}"
        )

        best_window = max(window_rows, key=lambda x: x["test_score"])

        window_results = []
        for row in window_rows:
            win = row["window"]
            window_results.append(
                WindowResult(
                    window_id=win.window_id,
                    train_start=win.train_start,
                    train_end=win.train_end,
                    val_start=win.val_start,
                    val_end=win.val_end,
                    test_start=win.test_start,
                    test_end=win.test_end,
                    genome=row["genome"],
                    train_metrics=row["train"].metrics.to_dict(),
                    val_metrics=row["val"].metrics.to_dict(),
                    test_metrics=row["test"].metrics.to_dict(),
                    train_score=row["train_score"],
                    val_score=row["val_score"],
                    test_score=row["test_score"],
                )
            )

        return ExperimentResult(
            best_genome=best_window["genome"],
            stitched_test_metrics=stitched_metrics,
            window_results=window_results,
            all_trials=pd.DataFrame(all_trials_rows),
            stitched_equity=stitched_eq,
            stitched_trades=stitched_trades,
        )
