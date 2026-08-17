"""冒烟测试：加载少量交易日，跑通环境与一次网络更新，并测量耗时。"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepscalper.agent import BDQAgent
from deepscalper.buffer import Transition
from deepscalper.config import Config
from deepscalper.data import load_days
from deepscalper.env import DayMarket, TradingEnv
from deepscalper.features import fit_feature_stats


def main():
    cfg = Config()
    t0 = time.time()
    days = load_days("301308", cfg.data_dir)
    print(f"load_days: {time.time()-t0:.1f}s, {len(days)} days")

    t0 = time.time()
    m = DayMarket(days[40], cfg)
    m.set_stats(fit_feature_stats([m], cfg) if cfg.normalize else None)
    print(f"DayMarket build: {time.time()-t0:.2f}s, n={m.n}, decisions={len(m.decision_points)}")

    agent = BDQAgent(cfg, seed=0)
    env = TradingEnv(m, cfg)
    obs = env.observation()
    print("obs shapes:", obs.micro_lob.shape, obs.private.shape, obs.macro.shape)

    # 交互 30 步并计时
    t0 = time.time()
    for i in range(30):
        action = agent.act(obs, epsilon=1.0)
        t_idx, priv_hist = env.step_idx, env.priv_hist.copy()
        res = env.step(action)
        agent.push(Transition(0, t_idx, action[0], action[1], res.train_reward,
                              res.t if not res.done else -1, res.done,
                              priv_hist, res.priv_hist, res.vol_label))
        if res.done:
            break
        obs = res.obs
    print(f"30 env steps: {time.time()-t0:.2f}s, nv={env.net_value():.4f}")

    # 填满 buffer 做一次更新并计时
    env = TradingEnv(m, cfg)
    obs = env.observation()
    t0 = time.time()
    while len(agent.buffer) < cfg.batch_size:
        action = agent.act(obs, epsilon=1.0)
        t_idx, priv_hist = env.step_idx, env.priv_hist.copy()
        res = env.step(action)
        agent.push(Transition(0, t_idx, action[0], action[1], res.train_reward,
                              res.t if not res.done else -1, res.done,
                              priv_hist, res.priv_hist, res.vol_label))
        if res.done:
            env = TradingEnv(m, cfg)
            obs = env.observation()
        else:
            obs = res.obs
    t0 = time.time()
    loss = agent.update([m], beta=0.4)
    print(f"one update (batch {cfg.batch_size}): {time.time()-t0:.3f}s loss={loss:.4e}")

    # 贪心前向计时
    t0 = time.time()
    for _ in range(20):
        agent.greedy(obs)
    print(f"20 greedy forwards: {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()
