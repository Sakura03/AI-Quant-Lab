def calculate_liquidation_price(is_long: bool, entry_price: float, leverage: int) -> float:
    """
        计算 USDT-M 合约的爆仓价（不考虑手续费）
    """

    if leverage <= 0:
        raise ValueError("杠杆倍数必须大于0")

    if is_long:
        # 多单爆仓价
        return entry_price * (1 - 1 / leverage)

    else:
        # 空单爆仓价
        return entry_price * (1 + 1 / leverage)
