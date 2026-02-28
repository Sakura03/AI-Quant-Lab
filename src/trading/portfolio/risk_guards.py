from __future__ import annotations


def drawdown_risk_scale(drawdown: float, dd_soft: float, dd_hard: float, min_risk_scale: float) -> float:
    dd = max(0.0, float(drawdown))
    if dd <= dd_soft:
        return 1.0
    if dd >= dd_hard:
        return float(min_risk_scale)
    span = max(dd_hard - dd_soft, 1e-12)
    alpha = (dd - dd_soft) / span
    return float(1.0 - alpha * (1.0 - min_risk_scale))


def concentration_penalty(max_symbol_share: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    if max_symbol_share <= threshold:
        return 0.0
    overflow = max_symbol_share - threshold
    return overflow / max(threshold, 1e-12)
