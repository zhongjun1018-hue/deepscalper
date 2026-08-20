"""交易成本：全项目显性费用的唯一来源（A 股线性参考值：双边佣金 + 卖出印花税）。

engine.run_day 的 grid_profit 是无量纲、不含费用的形态口径；fills_from_events 与
daily_net 把成交（买/卖价序列）折算为金额利润：买卖配对、期末残余敞口按最后有效
mid 盯市、逐笔扣显性费用。
"""

COMMISSION_RATE = 1e-4   # 双边佣金
STAMP_DUTY_RATE = 5e-4   # 卖出印花税


def fee_rate(side: int) -> float:
    """方向 side（+1 买 / −1 卖）的显性费率：买入仅佣金，卖出另加印花税。"""
    return COMMISSION_RATE + (STAMP_DUTY_RATE if side < 0 else 0.0)


def fee(side: int, notional: float) -> float:
    """成交额 notional 对应的显性费用。"""
    return notional * fee_rate(side)


def fills_from_events(events: list[dict]) -> list[tuple[int, float]]:
    """从 engine trace 事件中提取成交流 (side, price)：买 +1 / 卖 −1，保持原序。"""
    return [(+1 if e["kind"] == "buy" else -1, e["fill"])
            for e in events if e["kind"] in ("buy", "sell")]


def daily_net(fills: list[tuple[int, float]], close_mid: float) -> float:
    """单日净利润：Σ卖价 − Σ买价 + 残余敞口×close_mid − 逐笔显性费用。"""
    cash = 0.0
    exposure = 0
    for side, price in fills:
        exposure += side
        cash -= side * price
        cash -= price * fee_rate(side)
    return cash + exposure * close_mid
