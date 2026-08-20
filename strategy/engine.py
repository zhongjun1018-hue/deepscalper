"""报价驱动的网格逐日回放引擎：全项目网格策略执行口径的唯一来源。"""

import numpy as np

from strategy.grid import buy_crossed, sell_crossed


def run_day(bid1, ask1, mid, hard_exclude, width, *, decide_interval=1, trace=False):
    """在连续竞价快照序列上回放一张固定半宽网格。

    空仓且门控允许时在首个有效 mid 上以 mid 为中心开网；成交判定为对手方一档
    严格穿越边界（买一上穿上边界则卖出、卖一下穿下边界则买入，见 strategy/grid.py），
    成交价为边界价，成交后中心重置到成交价。日终不强制平仓。

    hard_exclude 与报价序列逐 tick 对齐（None 表示常开），只在净持仓为 0 时读取当拍
    信号：空仓期间每 decide_interval 个 tick 判一次；一旦成交，持仓途中的信号变化都不
    生效，直到敞口归零才立即重判，并从该 tick 重新计时。

    返回 buys / sells 与无量纲、不含费用的 grid_profit
    （特征标签与形态评分的定义口径；成本由 strategy/costs.py 叠加）。
    trace=True 时附带逐事件轨迹 events（webviz 依赖该结构）。
    """
    if len({len(bid1), len(ask1), len(mid)}) != 1:
        raise ValueError("bid1, ask1 and mid must have the same length")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("width must be a positive finite number")
    if hard_exclude is not None and len(hard_exclude) != len(mid):
        raise ValueError("hard_exclude must be None or aligned with the quote series")

    gated = hard_exclude is not None
    active = False
    enabled = not gated          # 无门控时恒放行，无需逐 tick 查询
    next_decision = 0
    center = upper = lower = entry_center = 0.0
    close_mid = float("nan")
    exposure = buys = sells = 0
    events = [] if trace else None

    def decide(i):
        # 两个调用点（空仓复判、成交归零）均已保证 exposure == 0
        nonlocal active, enabled, next_decision
        enabled = not bool(hard_exclude[i])
        if not enabled:
            active = False
        next_decision = i + decide_interval

    for i in range(len(mid)):
        bid = bid1[i]
        ask = ask1[i]
        anchor = mid[i]
        if np.isfinite(anchor):
            close_mid = float(anchor)
        opened = False

        if gated and exposure == 0 and i >= next_decision:
            decide(i)

        if exposure == 0 and enabled and not active and np.isfinite(anchor):
            center, active = float(anchor), True
            entry_center = center
            upper, lower = center + width, center - width
            opened = True
            if trace:
                events.append({
                    "index": int(i),
                    "kind": "open",
                    "center": center,
                    "upper": upper,
                    "lower": lower,
                    "exposure": exposure,
                })

        if active and not opened:
            kind = fill = None
            if np.isfinite(bid) and sell_crossed(bid, upper):
                kind, fill = "sell", upper
                sells, exposure, center = sells + 1, exposure - 1, upper
            elif np.isfinite(ask) and buy_crossed(ask, lower):
                kind, fill = "buy", lower
                buys, exposure, center = buys + 1, exposure + 1, lower

            if kind is not None:
                upper, lower = center + width, center - width
                if trace:
                    events.append({
                        "index": int(i),
                        "kind": kind,
                        "fill": float(fill),
                        "center": float(center),
                        "upper": float(upper),
                        "lower": float(lower),
                        "exposure": exposure,
                    })
                if gated and exposure == 0:
                    decide(i)

    trades = buys + sells
    grid_profit = 0.5 * (trades + exposure ** 2)
    if exposure:
        grid_profit += exposure * (close_mid - entry_center) / width
    result = {
        "buys": buys,
        "sells": sells,
        "grid_profit": float(grid_profit),
    }
    if trace:
        result["events"] = events
    return result
