"""门控点：从 webviz 日 JSON 重建引擎的实际停网段，并与常开网格在同一段的表现对照。

webviz 导出的 `excluded` 是识别信号（P>τ，按锚点前向填充）的区段，不是引擎状态：引擎只在
净持仓为 0 的锚点读信号，切换还需连续 confirm_n 拍一致（strategy.engine.run_day）。这里按
同一状态机在分钟网格上重放——敞口取门控网格的事件流，锚点是分钟末快照，分钟内成交先于判定——
得到逐分钟 enabled，其 False 连续段即停网段，每一段是一个门控点。常开网格在停网起点可能带仓
（exposure_in），对照时一并给出。
"""

from __future__ import annotations

import numpy as np

M = 237   # 压缩分钟数


def replay_enabled(day: dict, confirm_n: int) -> np.ndarray:
    """逐分钟 enabled (M,)；回放起点 t0 之前记 False。"""
    grid = day["grids"]["prediction"]
    t0 = day["t0"]
    adverse = np.zeros(M, dtype=bool)
    for start, end in grid["excluded"]:
        for m in range(int(start), min(int(end) + 1, M)):
            adverse[m] = start <= m + 0.95 <= end   # 锚点 x ≈ 分钟 + 0.95
    exposure = np.zeros(M, dtype=np.int64)
    for event in grid["events"]:
        exposure[event["minute"]:] = event["exposure"]

    enabled = np.zeros(M, dtype=bool)
    state, streak = True, 0
    for m in range(t0, M):
        if exposure[m] == 0:
            wanted = not adverse[m]
            if m == t0:
                state = wanted          # 日初自举不经连续确认
            elif wanted == state:
                streak = 0
            else:
                streak += 1
                if streak >= confirm_n:
                    state, streak = wanted, 0
        enabled[m] = state
    return enabled


def price_at(day: dict, minute: int) -> float:
    """该分钟末的最新价（分钟无快照时取其前最近一条）。"""
    x = np.asarray(day["x"])
    return float(day["price"][np.flatnonzero(x < minute + 1)[-1]])


def gating_points(day: dict, confirm_n: int) -> list[dict]:
    """门控点列表（停网段，含端点、按时间编号），附常开 / 门控两侧在段内的对照统计。

    对照口径：段内成交与敞口变化；`mtm_W` 为段内新成交按段末价盯市（买 p_end−f、卖 f−p_end，
    以 W_d 为单位）；`drift_W` 为段内价格位移 / W_d。
    """
    enabled = replay_enabled(day, confirm_n)
    width = day["width"]
    points = []
    m = day["t0"]
    while m < M:
        if enabled[m]:
            m += 1
            continue
        start = m
        while m < M and not enabled[m]:
            m += 1
        end = m - 1
        p_start, p_end = price_at(day, start), price_at(day, end)

        def side(key: str) -> dict:
            events = day["grids"][key]["events"]
            fills = [e for e in events if e["kind"] != "open" and start <= e["minute"] <= end]
            before = [e["exposure"] for e in events if e["minute"] < start]
            exposure_in = before[-1] if before else 0
            mtm = sum((p_end - e["fill"]) if e["kind"] == "buy" else (e["fill"] - p_end)
                      for e in fills) / width
            return {
                "fills": [{"minute": e["minute"], "kind": e["kind"], "fill": e["fill"]}
                          for e in fills],
                "buys": sum(e["kind"] == "buy" for e in fills),
                "sells": sum(e["kind"] == "sell" for e in fills),
                "exposure_in": exposure_in,
                "exposure_out": fills[-1]["exposure"] if fills else exposure_in,
                "mtm_W": round(mtm, 3),
            }

        always_on = side("none")
        points.append({
            "id": len(points) + 1, "start": start, "end": end, "minutes": end - start + 1,
            "price_start": p_start, "price_end": p_end,
            "drift_W": round((p_end - p_start) / width, 3),
            "always_on": always_on, "gated": side("prediction"),
        })
    return points
