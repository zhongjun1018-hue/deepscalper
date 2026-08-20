"""报价驱动的网格逐日回放引擎：全项目网格策略执行口径的唯一来源。"""

import numpy as np

from strategy.grid import buy_crossed, sell_crossed


def run_day(bid1, ask1, mid, hard_exclude, width, *, anchors=None, confirm_n=2,
            trace=False):
    """在连续竞价快照序列上回放一张固定半宽网格。

    空仓且门控允许时在首个有效 mid 上以 mid 为中心开网；成交判定为对手方一档
    严格穿越边界（买一上穿上边界则卖出、卖一下穿下边界则买入，见 strategy/grid.py），
    成交价为边界价，成交后中心重置到成交价。日终不强制平仓。

    撮合逐 tick 进行，中心只随成交移动。anchors 为分钟锚点的 tick 索引数组（与报价
    序列同一索引系，data_provider/ticks.py 的分钟网格），只给出门控的判定节奏：
    门控（hard_exclude 与报价序列逐 tick 对齐，None 表示常开）只在净持仓为 0 时
    读取，空仓期间在锚点 tick 复判；状态切换（开↔关）要求连续 confirm_n 个判定
    一致才生效（去抖），单拍判定只更新计数器；敞口归零即刻重判并重置确认计数。
    anchors=None 时门控空仓期间逐 tick 复判（窗口网格特征的反事实回放用）。

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

    is_anchor = np.zeros(len(mid), dtype=bool)
    if anchors is not None:
        is_anchor[np.asarray(anchors, dtype=np.int64)] = True

    gated = hard_exclude is not None
    active = False
    enabled = None if gated else True   # 无门控时恒放行；门控下首个判定直接定初态
    streak = 0                   # 与当前状态相反的连续判定数（连续确认去抖）
    center = upper = lower = entry_center = 0.0
    close_mid = float("nan")
    exposure = buys = sells = 0
    events = [] if trace else None

    def judge(i):
        # 两个调用点（空仓锚点复判、成交归零）均已保证 exposure == 0
        nonlocal active, enabled, streak
        wanted = not bool(hard_exclude[i])
        if enabled is None:      # 日初自举不是状态切换，不经连续确认
            enabled = wanted
            return
        if wanted == enabled:
            streak = 0
            return
        streak += 1
        if streak >= confirm_n:
            enabled, streak = wanted, 0
            if not enabled:
                active = False

    def record(i, kind, **extra):
        events.append({"index": int(i), "kind": kind, "center": center,
                       "upper": upper, "lower": lower, "exposure": exposure, **extra})

    for i in range(len(mid)):
        bid = bid1[i]
        ask = ask1[i]
        anchor = mid[i]
        if np.isfinite(anchor):
            close_mid = float(anchor)
        opened = False

        if gated and exposure == 0 and (is_anchor[i] or anchors is None):
            judge(i)

        if exposure == 0 and enabled and not active and np.isfinite(anchor):
            center, active = float(anchor), True
            entry_center = center
            upper, lower = center + width, center - width
            opened = True
            if trace:
                record(i, "open")

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
                    record(i, kind, fill=float(fill))
                if gated and exposure == 0:
                    streak = 0   # 敞口归零：重置确认计数并即刻重判
                    judge(i)

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
