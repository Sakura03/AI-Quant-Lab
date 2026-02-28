from types import SimpleNamespace

from trading.optimizer import objective_score


def test_objective_score_penalizes_drawdown_breach():
    m = SimpleNamespace(sharpe=1.5, max_drawdown=0.35, turnover=2.0)
    s = objective_score(
        m,
        drawdown_limit=0.2,
        weights={"sharpe": 1.0, "max_drawdown_penalty": 2.0, "turnover_penalty": 0.1},
    )
    assert s == -1e9
