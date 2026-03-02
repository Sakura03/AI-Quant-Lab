from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionSimulator:
    """Apply trading frictions (slippage and fees) to simulated fills."""

    fee_rate: float
    slippage_bps: float

    def apply_slippage(self, raw_price: float, order_side: str) -> float:
        """Shift raw price by configured slippage for buy/sell direction."""
        if order_side not in {"buy", "sell"}:
            raise ValueError(f"Unknown order_side: {order_side}")
        ratio = self.slippage_bps / 10000.0
        if order_side == "buy":
            return raw_price * (1.0 + ratio)
        return raw_price * (1.0 - ratio)

    def calc_fee(self, price: float, qty: float) -> float:
        """Compute trading fee from notional."""
        return abs(price * qty) * self.fee_rate


def side_to_open_order(side: int) -> str:
    """Map position side to open order direction."""
    return "buy" if side > 0 else "sell"


def side_to_close_order(side: int) -> str:
    """Map position side to close order direction."""
    return "sell" if side > 0 else "buy"
