import numpy as np

from trading.strategies.factory import STRATEGY_REGISTRY, build_strategy, mutate_strategy_params, sample_strategy_params


def test_factory_build_sample_mutate_for_all_strategies():
    rng = np.random.default_rng(42)
    for sid in STRATEGY_REGISTRY:
        params = sample_strategy_params(sid, rng)
        strategy = build_strategy(sid, params)
        assert strategy.strategy_id == sid
        mutated = mutate_strategy_params(sid, params, 0.2, rng)
        assert isinstance(mutated, dict)
        assert mutated
