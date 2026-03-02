from __future__ import annotations

import argparse
import copy
import os.path as osp
from datetime import datetime

import yaml

from trading.backtest.engine import BacktestEngine
from trading.config import TradingConfig, parse_timestamp
from trading.optimize.evolver import WalkForwardEvolver
from trading.report.writer import build_run_dir, write_backtest_report, write_optimizer_report


def parse_args() -> argparse.Namespace:
    """Parse trading CLI arguments."""
    parser = argparse.ArgumentParser(description="Enterprise multi-strategy crypto trading framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bt = sub.add_parser("backtest", help="Run one backtest")
    p_bt.add_argument("--config", required=True, help="Path to config YAML")
    p_bt.add_argument("--signal-tf", help="Signal timeframe override")
    p_bt.add_argument("--regime-tf", help="Regime timeframe override")
    p_bt.add_argument("--strategies", help="Comma-separated strategy IDs override")
    p_bt.add_argument("--genome-file", help="Path to optimizer best_genome.yml")
    p_bt.add_argument("--start", help="Start timestamp override")
    p_bt.add_argument("--end", help="End timestamp override")
    p_bt.add_argument("--quiet", action="store_true", help="Disable progress logs")

    p_opt = sub.add_parser("optimize", help="Run walk-forward evolutionary optimization")
    p_opt.add_argument("--config", required=True, help="Path to config YAML")
    p_opt.add_argument("--quiet", action="store_true", help="Disable progress logs")

    return parser.parse_args()


def _config_with_optional_range(cfg: TradingConfig, start: str | None, end: str | None) -> TradingConfig:
    """Clone config and optionally override data start/end."""
    if not start and not end:
        return cfg
    raw = copy.deepcopy(cfg.to_dict())
    if start:
        raw["data"]["start"] = parse_timestamp(start).strftime("%Y%m%d-%H%M%S")
    if end:
        raw["data"]["end"] = parse_timestamp(end).strftime("%Y%m%d-%H%M%S")
    return TradingConfig.from_dict(raw)


def _load_genome(path: str) -> dict:
    """Load optimizer genome file and validate required keys."""
    if not osp.isfile(path):
        raise FileNotFoundError(f"Genome file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        genome = yaml.safe_load(f)
    if not isinstance(genome, dict):
        raise TypeError("Genome file root must be a mapping")
    if "strategy_ids" not in genome or "strategy_params" not in genome:
        raise ValueError("Genome file must include strategy_ids and strategy_params")
    return genome


def _make_monitor(enabled: bool):
    """Build a timestamped progress logger callback."""
    if not enabled:
        return None

    def _monitor(msg: str):
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {msg}", flush=True)

    return _monitor


def run_backtest(args: argparse.Namespace):
    """CLI handler for single backtest mode."""
    monitor = _make_monitor(not args.quiet)
    if monitor is not None:
        monitor("Loading backtest config")

    cfg = TradingConfig.parse_file(args.config)
    cfg = _config_with_optional_range(cfg, args.start, args.end)

    genome = _load_genome(args.genome_file) if args.genome_file else None

    signal_tf = cfg.data.signal_tfs[0]
    regime_tf = cfg.data.regime_tfs[0] if cfg.data.regime_tfs else None
    strategies = list(cfg.strategies.enabled)
    strategy_params = {sid: {} for sid in strategies}

    if genome is not None:
        signal_tf = str(genome.get("signal_tf", signal_tf))
        regime_tf = genome.get("regime_tf", regime_tf)
        strategies = [str(s) for s in genome.get("strategy_ids", strategies)]
        raw_params = genome.get("strategy_params", {})
        strategy_params = {sid: dict(raw_params.get(sid, {})) for sid in strategies}

    if args.signal_tf:
        signal_tf = args.signal_tf
    if args.regime_tf is not None:
        regime_tf = args.regime_tf
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        strategy_params = {sid: dict(strategy_params.get(sid, {})) for sid in strategies}

    if monitor is not None:
        monitor(
            f"Starting backtest | signal_tf={signal_tf} regime_tf={regime_tf} "
            f"strategies={','.join(strategies)} symbols={len(cfg.data.universe)}"
        )

    engine = BacktestEngine(
        config=cfg,
        signal_tf=signal_tf,
        regime_tf=regime_tf,
        strategy_ids=strategies,
        strategy_params=strategy_params,
        progress_cb=monitor,
        progress_name="backtest",
    )
    result = engine.run()

    out_dir = build_run_dir(cfg.output.dir, "backtest")
    write_backtest_report(result, cfg, out_dir)

    m = result.metrics
    print(f"Saved to: {out_dir}")
    print(
        f"SignalTF={signal_tf} | RegimeTF={regime_tf} | Strategies={','.join(strategies)} | "
        f"Sharpe={m.sharpe:.3f} | MaxDD={m.max_drawdown*100:.2f}% | AnnRet={m.annual_return*100:.2f}% | Trades={m.trade_count}"
    )


def run_optimize(args: argparse.Namespace):
    """CLI handler for walk-forward optimization mode."""
    monitor = _make_monitor(not args.quiet)
    if monitor is not None:
        monitor("Loading optimize config")

    cfg = TradingConfig.parse_file(args.config)
    evo = WalkForwardEvolver(cfg, progress_cb=monitor)
    if monitor is not None:
        monitor("Starting walk-forward optimization")
    result = evo.run()

    out_dir = build_run_dir(cfg.output.dir, "optimize")
    write_optimizer_report(result, cfg, out_dir)

    m = result.stitched_test_metrics
    print(f"Saved to: {out_dir}")
    print(
        f"Stitched OOS Sharpe={m.get('sharpe', 0.0):.3f} | "
        f"MaxDD={m.get('max_drawdown', 0.0)*100:.2f}% | AnnRet={m.get('annual_return', 0.0)*100:.2f}% | "
        f"Trades={m.get('trade_count', 0)}"
    )


def main():
    """CLI entrypoint dispatcher."""
    args = parse_args()
    if args.command == "backtest":
        run_backtest(args)
        return
    if args.command == "optimize":
        run_optimize(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
