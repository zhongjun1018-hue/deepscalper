"""成本叠加回测：在 engine 的无费用成交流之上按 strategy/costs.py 计费。

engine.run_day 的 grid_profit 是无量纲、不含费用的形态口径；本模块把成交
（买/卖价序列）折算为金额利润：买卖配对、期末残余敞口按最后有效 mid 盯市、
逐笔扣显性费用，输出逐日净利润与汇总。
"""

from __future__ import annotations

import numpy as np

from strategy.costs import fee_rate


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


def backtest(days: list[tuple[list[tuple[int, float]], float]]) -> dict:
    """逐日净利润与汇总。

    days 为逐日 (fills, close_mid)：fills 是 (side, price) 成交流（可由
    fills_from_events 从 trace 事件得到），close_mid 为当日最后一个有效 mid。
    返回 {"daily": (D,) 逐日净利润, "total", "mean", "std"}。
    """
    daily = np.array([daily_net(fills, close_mid) for fills, close_mid in days],
                     dtype=np.float64)
    return {
        "daily": daily,
        "total": float(daily.sum()),
        "mean": float(daily.mean()) if len(daily) else float("nan"),
        "std": float(daily.std(ddof=0)) if len(daily) else float("nan"),
    }
