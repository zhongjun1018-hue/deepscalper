"""冒烟测试（python -m control.smoke_test [标的]）：加载真实交易日与统一缓存，
跑通定长决策的网格环境与一次网络更新，并测量耗时。"""

import sys
import time

from data_provider.split import chronological_split
from data_provider.ticks import load_days
from data_provider.windows import load_cache

from .agent import BranchQAgent
from .baselines import run_fixed_grid
from .buffer import Transition
from .config import Config
from .env import StepResult, TradingEnv, action_params
from .features import fit_feature_stats
from .train import build_markets


def interact(env: TradingEnv, cfg: Config, agent: BranchQAgent, day_id: int = 0) -> StepResult:
    """交互一步并入队，返回环境结果。"""
    obs = env.observation()
    action = agent.act(obs, epsilon=1.0)
    minute, priv_hist = env.minute, env.priv_window(env.minute)
    res = env.step(action_params(cfg, action))
    agent.push(Transition(day_id, minute, action, res.train_reward,
                          res.minute if not res.done else -1, res.done,
                          priv_hist, res.priv_hist))
    return res


def main(symbol: str):
    cfg = Config(symbols=(symbol,))
    t0 = time.time()
    days = load_days(symbol, cfg.data_dir, cfg.window.atr_window)
    print(f"load_days({symbol}): {time.time()-t0:.1f}s, {len(days)} days")

    split = chronological_split([d.date for d in days])
    train_days = [d for d in days if d.date in set(split.train)]
    print(f"split: train {len(train_days)} / val {len(split.val)} / test {len(split.test)}")

    # 预测块只读不重训（缺失时按零特征读取，与 control.train 同口径）
    t0 = time.time()
    cache = load_cache(symbol, data_dir=cfg.data_dir, cache_dir=cfg.cache_dir,
                       spec=cfg.window, zero_nan=True)
    print(f"unified cache: {time.time()-t0:.1f}s, {len(cache['dates'])} days")

    # 仅用训练集拟合标准化统计量（与 control.train 同一口径）
    t0 = time.time()
    train_m = build_markets(train_days, cfg, cache, symbol_id=0)
    stats = fit_feature_stats(train_m, cfg) if cfg.normalize else None
    for m in train_m:
        m.set_stats(stats)
    m = train_m[len(train_m) // 2]
    print(f"build_markets + fit stats: {time.time()-t0:.1f}s, "
          f"train_days={len(train_m)}, window={m.window.shape}")

    agent = BranchQAgent(cfg, seed=0)
    env = TradingEnv(m)
    obs = env.observation()
    print("obs shapes:", obs.micro_lob.shape, obs.private.shape, obs.macro.shape)

    # 随机策略跑完一天并计时（起点偏移取决策间隔的一半，顺带覆盖随机起点路径）
    t0 = time.time()
    reward_sum = 0.0
    env = TradingEnv(m, start_offset_min=cfg.decision_interval_min // 2)
    while True:
        res = interact(env, cfg, agent)
        reward_sum += res.reward
        if res.done:
            break
    accounting_error = reward_sum - (env.net_value() - 1.0)
    assert abs(accounting_error) < 1e-10
    print(f"one random day: {time.time()-t0:.2f}s, steps={env.n_steps}, "
          f"fills={len(env.fills)}, nv={env.net_value():.4f}, "
          f"accounting_error={accounting_error:.1e}")

    # 填满 buffer 做一次更新并计时
    while len(agent.buffer) < cfg.batch_size:
        env = TradingEnv(m)
        while True:
            if interact(env, cfg, agent).done:
                break
    t0 = time.time()
    q_loss = agent.update([m], beta=0.4)
    print(f"one update (batch {cfg.batch_size}): {time.time()-t0:.3f}s "
          f"q_loss={q_loss:.4e}")

    # 固定半宽网格基线与贪心前向计时
    t0 = time.time()
    fixed = run_fixed_grid({symbol: [m]}, width=0.1)
    diagnostics = fixed["per_symbol"][symbol]["diagnostics"]
    print(f"fixed grid (h=0.1): {time.time()-t0:.2f}s TR={fixed['TR']:.4f} "
          f"fills={diagnostics['n_fills']:.0f} "
          f"decisions={diagnostics['n_decisions']:.0f}")

    t0 = time.time()
    for _ in range(20):
        agent.greedy(env.observation())
    print(f"20 greedy forwards: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "301308")
