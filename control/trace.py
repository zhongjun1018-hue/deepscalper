"""RL 决策轨迹回放：加载训练检查点，对单个交易日贪心回放并记录网格与成交（webviz 用）。

只记录环境真实提供的信息：决策点的生效网格按 env.step 的口径计算（决策点发生立即
成交时中心已移至成交价，触发线按新中心重算），网格成交取自 env.fills；平仓 / 日终
扫单不进入 env.fills，按同一请求量在同快照上重放纯函数 market.sweep 补记（逐档均价），
数量与环境的持仓记账一致。
"""

from __future__ import annotations

import torch

from strategy.grid import boundaries, half_width

from .config import Config
from .env import DayMarket, TradingEnv, action_params
from .model import BDQNetwork


def load_checkpoint(path: str, device) -> tuple[BDQNetwork, Config]:
    """加载 control/train.py save_checkpoint 保存的检查点：重建网络并加载权重（eval 模式）。"""
    payload = torch.load(path, map_location=device)
    cfg = Config(**payload["config"])
    net = BDQNetwork(cfg).to(device).eval()
    net.load_state_dict(payload["state_dict"])
    return net, cfg


def _sweep_fill(market: DayMarket, t: int, qty: float) -> dict | None:
    """重放 market.sweep（纯函数）补记一笔扫单成交：{t, side, price(逐档均价), qty}。"""
    filled, cash_delta, fee_cost, _ = market.sweep(t, qty)
    if filled == 0.0:   # sweep 的首个返回值带符号（正买负卖），零才是不成交
        return None
    side = 1.0 if qty > 0 else -1.0
    notional = -side * (cash_delta + fee_cost)   # sweep 记账：cash_delta = −side·成交额 − fee
    return {"t": t, "side": "buy" if side > 0 else "sell",
            "price": notional / abs(filled), "qty": abs(filled)}


def trace_day(market: DayMarket, policy) -> dict:
    """对单个交易日回放 policy，记录每个决策点的生效网格与全部成交。

    policy(obs) → 动作档位 (半宽档, 数量档)，由调用方用 load_checkpoint 的网络构造
    （贪心策略）。返回：
      {"decisions": [{t, width, size, center, upper, lower}],  # 各决策点的生效网格
       "fills": [{t, side, price, qty}],                        # 全部成交（含立即成交与平仓扫单）
       "log": episode_log 摘要}
    平仓档（width == 0）不建网格，upper / lower 记 None、size 记 0（与 env.step 一致）。
    """
    env = TradingEnv(market, hindsight=False)
    base = env.cfg.base_position
    obs = env.observation()
    decisions, fills = [], []
    n_fills = 0
    while True:
        action = policy(obs)
        params = action_params(env.cfg, action)
        t, pos_before, center = env.t, env.pos, env.center
        res = env.step(params)

        new_fills = env.fills[n_fills:]
        n_fills = len(env.fills)
        if params.width > 0.0:
            # 决策点发生立即成交时中心已移至成交价，环境按新中心重算触发线（env.step）
            immediate = next((f for f in new_fills if f.immediate), None)
            if immediate is not None:
                center = immediate.price
            hw = half_width(params.width, market.atr, center,
                            env.cfg.window.min_width_ratio)
            upper, lower = boundaries(center, hw, env.cfg.tick_size)
            size = params.size
        else:
            upper = lower = None
            size = 0                       # 平仓档不建网格，生效数量记 0
        decisions.append({"t": t, "width": params.width, "size": size,
                          "center": center, "upper": upper, "lower": lower})

        for f in new_fills:
            fills.append({"t": f.tick, "side": "buy" if f.qty > 0 else "sell",
                          "price": f.price, "qty": abs(f.qty)})
        pos_walk = pos_before + sum(f.qty for f in new_fills)
        if params.width == 0.0 and pos_before != base:
            record = _sweep_fill(market, t, -(pos_before - base))   # 决策点平仓（3.5）
            if record is not None:
                fills.append(record)
                pos_walk += record["qty"] if record["side"] == "buy" else -record["qty"]
        if res.done and pos_walk != base:
            record = _sweep_fill(market, res.t, -(pos_walk - base))  # 日终平回底仓（3.5）
            if record is not None:
                fills.append(record)
        if res.done:
            break
        obs = res.obs
    return {"decisions": decisions, "fills": fills, "log": env.episode_log()}
