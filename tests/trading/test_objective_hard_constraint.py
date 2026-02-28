from trading.domain.types import Metrics
from trading.evaluate.metrics import score_metrics


def test_score_returns_extreme_negative_when_drawdown_breaches_limit():
    train = Metrics(sharpe=1.5, max_drawdown=0.1, turnover=1.0, trade_count=100)
    val = Metrics(sharpe=1.2, max_drawdown=0.2, turnover=1.2, trade_count=100)

    score = score_metrics(
        train_metrics=train,
        eval_metrics=val,
        max_symbol_share=0.2,
        objective_weights={
            "sharpe": 1.0,
            "max_drawdown": 2.0,
            "turnover": 0.1,
            "overfit_gap": 0.2,
            "concentration": 0.5,
        },
        hard_limits={"max_drawdown": 0.15},
        min_trades=10,
    )

    assert score == -1e9
