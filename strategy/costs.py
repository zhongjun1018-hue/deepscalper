"""交易成本：全项目显性费用的唯一来源（A 股线性参考值：双边佣金 + 卖出印花税）。"""

COMMISSION_RATE = 1e-4   # 双边佣金
STAMP_DUTY_RATE = 5e-4   # 卖出印花税


def fee_rate(side: int) -> float:
    """方向 side（+1 买 / −1 卖）的显性费率：买入仅佣金，卖出另加印花税。"""
    return COMMISSION_RATE + (STAMP_DUTY_RATE if side < 0 else 0.0)


def fee(side: int, notional: float) -> float:
    """成交额 notional 对应的显性费用。"""
    return notional * fee_rate(side)
