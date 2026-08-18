"""冒烟测试：加载少量交易日，跑通网格环境与一次网络更新，并测量耗时。"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridscalper.agent import BDQAgent
from gridscalper.baselines import run_fixed_grid
from gridscalper.buffer import Transition
from gridscalper.config import Config
from gridscalper.data import load_days
from gridscalper.env import DayMarket, StepResult, TradingEnv, action_params
from gridscalper.features import fit_feature_stats


def interact(env: TradingEnv, cfg: Config, agent: BDQAgent, day_id: int = 0) -> StepResult:
    """交互一步并入队，返回环境结果。"""
    obs = env.observation()
    action = agent.act(obs, epsilon=1.0)
    t, priv_hist = env.t, env.priv_window(env.t)
    res = env.step(action_params(cfg, action))
    agent.push(Transition(day_id, t, action, res.train_reward, res.tau,
                          res.t if not res.done else -1, res.done,
                          priv_hist, res.priv_hist, res.vol_label))
    return res


def main(symbol: str):
    cfg = Config()
    t0 = time.time()
    days = load_days(symbol, cfg.data_dir, cfg.atr_days)
    print(f"load_days({symbol}): {time.time()-t0:.1f}s, {len(days)} days")

    t0 = time.time()
    m = DayMarket(days[40], cfg)
    m.set_stats(fit_feature_stats([m], cfg) if cfg.normalize else None)
    print(f"DayMarket build: {time.time()-t0:.2f}s, n={m.n}, atr={m.atr:.3f}, p0={m.p0:.2f}")

    agent = BDQAgent(cfg, seed=0)
    env = TradingEnv(m)
    obs = env.observation()
    print("obs shapes:", obs.micro_lob.shape, obs.private.shape, obs.macro.shape)

    # 随机策略跑完一天并计时
    t0 = time.time()
    reward_sum = 0.0
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
    q_loss, vol_loss = agent.update([m], beta=0.4)
    print(f"one update (batch {cfg.batch_size}): {time.time()-t0:.3f}s "
          f"q_loss={q_loss:.4e} vol_loss={vol_loss:.4e}")

    # 固定半宽网格基线与贪心前向计时
    t0 = time.time()
    fixed = run_fixed_grid([m], half_width=0.1)
    print(f"fixed grid (h=0.1): {time.time()-t0:.2f}s TR={fixed['TR']:.4f} "
          f"fills={fixed['diagnostics']['n_fills']:.0f} "
          f"mean_tau={fixed['diagnostics']['mean_tau']:.1f}")

    t0 = time.time()
    for _ in range(20):
        agent.greedy(env.observation())
    print(f"20 greedy forwards: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "301308")
