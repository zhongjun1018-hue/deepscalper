"""RL 检查点工具与决策轨迹回放：解析 / 加载统一训练检查点、构建贪心策略与测试段
市场（webviz 与统一回测共用），并对单个交易日贪心回放记录网格与成交（webviz 用）。

只记录环境真实提供的信息：决策点的生效网格按 env.step 的口径计算（决策点发生立即
成交时中心已移至成交价，触发线按新中心重算），成交直接取自 env.fills——网格成交、
决策点平仓与日终清仓都在其中，平仓与清仓按对手方一档价成交（control/env.py）。
"""

from __future__ import annotations

import glob
import os

import torch

from data_provider.windows import WindowSpec
from strategy.grid import boundaries, half_width

from .config import Config
from .env import DayMarket, TradingEnv, action_params
from .features import FeatureStats
from .model import BranchQNetwork, to_batch


def resolve_checkpoint(method: str = "GRID", seed: int = 0,
                       w: float | None = None, lam: float | None = None,
                       checkpoint: str | None = None) -> str:
    """检查点路径：checkpoint 显式给定，否则按 control/runs 的结果命名规则解析。

    完整名为 <method>_w<w>_lam<λ>_seed<s>.pt（w/λ 缺省取 control Config 默认值）；
    control.train 的命名规则是不适用的超参不在文件名中（如 GRID-NH 无 w 标签），
    完整名未命中时退到 <method>*_seed<s>.pt 的唯一匹配。不唯一抛 ValueError，
    未找到抛 FileNotFoundError。
    """
    if checkpoint:
        path = checkpoint
    else:
        defaults = Config()
        w = defaults.hindsight_weight if w is None else w
        lam = defaults.inventory_lambda if lam is None else lam
        path = os.path.join(defaults.runs_dir, f"{method}_w{w:g}_lam{lam:g}_seed{seed}.pt")
        if not os.path.exists(path):
            matches = sorted(glob.glob(
                os.path.join(defaults.runs_dir, f"{method}_*_seed{seed}.pt")))
            if len(matches) == 1:
                path = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"{method} 检查点不唯一："
                                 + "、".join(os.path.basename(m) for m in matches)
                                 + "；请用 --w/--lam 或 --checkpoint 明确指定。")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 RL 检查点：{path}。"
            "请先运行 python -m control.train 完成训练（断点续跑会复用已有产物），"
            "或用 --checkpoint 显式指定检查点路径。")
    return path


def load_checkpoint(path: str, device) -> tuple[BranchQNetwork, Config, FeatureStats | None]:
    """加载 control/train.py save_checkpoint 保存的检查点（eval 模式）。

    返回（网络, 配置, 逐标的标准化统计量）；统计量随检查点保存，回放侧不重新拟合
    （prepare_test_markets）。
    """
    payload = torch.load(path, map_location=device)
    config = dict(payload["config"])
    # save_checkpoint 以 asdict 序列化配置，嵌套的 WindowSpec 需从 dict 还原
    config["window"] = WindowSpec(**config["window"])
    cfg = Config(**config)
    net = BranchQNetwork(cfg).to(device).eval()
    net.load_state_dict(payload["state_dict"])
    stats = (FeatureStats.from_state_dict(payload["feature_stats"])
             if payload["feature_stats"] is not None else None)
    return net, cfg, stats


def greedy_policy(net: BranchQNetwork, device):
    """由检查点网络构建贪心档位策略 policy(obs) → (半宽档, 数量档)。

    平仓档只在净持仓非零时可选；数量单档时网络无数量分支，数量档恒为 0
    （均与 BranchQAgent.greedy 同一口径）。
    """
    def policy(obs) -> tuple[int, int]:
        with torch.no_grad():
            q = net(*to_batch([obs], device))
        if not obs.flatten_allowed:
            q[0][:, 0] = -torch.inf
        gears = [int(branch.argmax(-1).item()) for branch in q]
        return gears[0], gears[1] if len(gears) > 1 else 0

    return policy


def prepare_test_markets(symbol: str, cfg: Config, stats: FeatureStats | None,
                         data_dir: str = "data",
                         cache_dir: str = "cache") -> list[DayMarket]:
    """按 7:1:2 切分构建一个标的的测试段回放市场并挂载检查点的标准化统计量。

    统计量来自检查点（统一训练逐标的在训练段拟合，无前视泄漏）；symbol_id 与
    forecast 同口径：排序后标的集合中的索引，也是统计量的行索引。标的须在检查点
    的训练集合中——embedding 与标准化统计量都按该集合定义，集合外无从回放。
    """
    from data_provider.split import chronological_split
    from data_provider.ticks import load_days
    from data_provider.windows import load_cache

    from .train import build_markets

    if symbol not in cfg.symbols:
        raise ValueError(f"标的 {symbol} 不在检查点的训练集合中，无法回放"
                         "（symbol embedding 与标准化统计量按训练集合定义）。")
    days = load_days(symbol, data_dir, cfg.window.atr_window)
    split = chronological_split([d.date for d in days])
    test_days = [d for d in days if d.date in set(split.test)]
    cache = load_cache(symbol, data_dir=data_dir, cache_dir=cache_dir,
                       spec=cfg.window, zero_nan=True)
    symbol_id = sorted(cfg.symbols).index(symbol)
    test_markets = build_markets(test_days, cfg, cache, symbol_id)
    for market in test_markets:
        market.set_stats(stats)
    return test_markets


def trace_day(market: DayMarket, policy) -> dict:
    """对单个交易日回放 policy，记录每个决策点的生效网格与全部成交。

    policy(obs) → 动作档位 (半宽档, 数量档)，由调用方以 greedy_policy 构造。返回：
      {"decisions": [{t, width, size, center, upper, lower}],  # 各决策点的生效网格
       "fills": [{t, side, price, qty, kind}],                  # 全部成交（含平仓与清仓，kind 见 Fill）
       "ret": 相对底仓的超额收益（与 control.train.replay_day 同口径）,
       "log": episode_log 摘要}
    平仓档（width == 0）不建网格，upper / lower 记 None、size 记 0（与 env.step 一致）。
    """
    env = TradingEnv(market, hindsight=False)
    obs = env.observation()
    decisions = []
    while True:
        action = policy(obs)
        params = action_params(env.cfg, action)
        t, center = env.t, env.center
        n_fills = len(env.fills)
        res = env.step(params)

        if params.width > 0.0:
            # 决策点发生立即成交时中心已移至成交价，环境按新中心重算触发线（env.step）
            immediate = next((f for f in env.fills[n_fills:] if f.kind == "immediate"), None)
            if immediate is not None:
                center = immediate.price
            hw = half_width(params.width, market.atr, market.pre_close,
                            env.cfg.window.min_width_ratio)
            upper, lower = boundaries(center, hw, env.cfg.tick_size)
            size = params.size
        else:
            upper = lower = None
            size = 0                       # 平仓档不建网格，生效数量记 0
        decisions.append({"t": t, "width": params.width, "size": size,
                          "center": center, "upper": upper, "lower": lower})
        if res.done:
            break
        obs = res.obs
    fills = [{"t": f.tick, "side": "buy" if f.qty > 0 else "sell",
              "price": f.price, "qty": abs(f.qty), "kind": f.kind} for f in env.fills]
    return {"decisions": decisions, "fills": fills,
            "ret": env.net_value() - 1.0, "log": env.episode_log()}
