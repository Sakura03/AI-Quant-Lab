from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionModel:
    """成交成本模型，封装滑点与手续费计算。"""
    fee_rate: float
    slippage_bps: float

    def apply_slippage(self, raw_price: float, order_side: str) -> float:
        """根据买卖方向对原始价格施加滑点，返回模拟成交价。"""
        if order_side not in {"buy", "sell"}:
            raise ValueError(f"未知 order_side: {order_side}")
        ratio = self.slippage_bps / 10000.0
        if order_side == "buy":
            return raw_price * (1.0 + ratio)
        return raw_price * (1.0 - ratio)

    def calc_fee(self, price: float, qty: float) -> float:
        """按名义成交额计算手续费。"""
        return abs(price * qty) * self.fee_rate


def side_to_open_order_side(side: int) -> str:
    """将持仓方向映射为开仓订单方向。"""
    return "buy" if side > 0 else "sell"


def side_to_close_order_side(side: int) -> str:
    """将持仓方向映射为平仓订单方向。"""
    return "sell" if side > 0 else "buy"
