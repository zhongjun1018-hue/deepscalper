"""DQN 基线（论文 5.3）：3 动作（做空 / 持有 / 做多）市价单、固定数量。

状态为平铺特征（宏观 12 + 末帧微观 50 + 私有 3），网络为 MLP，
均匀经验回放 + 一步 TD。与 DeepScalper 共用环境与评估口径。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .env import DayMarket, TradingEnv
from .metrics import financial_metrics

# 动作 → (价位偏移索引, 数量索引)：±5 档相对对手价的穿价档保证市价成交
ACTION_MAP = {0: (0, 0), 1: (5, 4), 2: (10, 8)}  # short / hold / long
STATE_DIM = 12 + 50 + 3


class _QNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)


def _state(m: DayMarket, t: int, pos: float, cash: float) -> np.ndarray:
    obs = m.observe(t, pos, cash, m.cash0)
    return np.concatenate([obs.macro, m.micro[t], obs.private[0]]).astype(np.float32)


def train_dqn(
    cfg: Config,
    markets: list[DayMarket],
    val_markets: list[DayMarket],
    seed: int,
    log_prefix: str = "[DQN]",
) -> tuple[_QNet, dict]:
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    online, target = _QNet(), _QNet()
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    replay: list[tuple] = []
    updates = 0

    best_val_tr, best_state = -np.inf, None
    for epoch in range(1, cfg.epochs + 1):
        for day_id, m in enumerate(markets):
            env = TradingEnv(m, cfg, hindsight=False)
            env.reset()
            eps = cfg.epsilon_at(epoch, day_id / max(1, len(markets)))
            while True:
                s = _state(m, env.t, env.pos, env.cash)
                a = int(rng.integers(3)) if rng.random() < eps else int(online(torch.as_tensor(s)).argmax().item())
                res = env.step(ACTION_MAP[a])
                if res.done:
                    replay.append((s, a, res.reward, None))
                    break
                s2 = _state(m, env.t, env.pos, env.cash)
                replay.append((s, a, res.reward, s2))
                if len(replay) > cfg.buffer_capacity:
                    replay.pop(0)
                if len(replay) >= cfg.batch_size and env.step_idx % cfg.update_every == 0:
                    idx = rng.choice(len(replay), cfg.batch_size, replace=True)
                    batch = [replay[i] for i in idx]
                    S = torch.as_tensor(np.stack([b[0] for b in batch]))
                    A = torch.as_tensor([b[1] for b in batch])
                    R = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
                    nonterm = [b[3] for b in batch]
                    with torch.no_grad():
                        nq = torch.zeros(len(batch))
                        keep = [i for i, x in enumerate(nonterm) if x is not None]
                        if keep:
                            nq[keep] = target(torch.as_tensor(np.stack([nonterm[i] for i in keep]))).max(-1).values
                        y = R + cfg.gamma * nq
                    loss = nn.functional.mse_loss(online(S).gather(1, A[:, None]).squeeze(1), y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    updates += 1
                    if updates % cfg.target_sync == 0:
                        target.load_state_dict(online.state_dict())

        val = evaluate_dqn(online, val_markets, cfg)
        print(f"{log_prefix} epoch {epoch}/{cfg.epochs} val_TR={val['TR']:.4f} val_SR={val['SR']:.3f}", flush=True)
        if val["TR"] > best_val_tr:
            best_val_tr, best_state = val["TR"], {k: v.clone() for k, v in online.state_dict().items()}

    if best_state is not None:
        online.load_state_dict(best_state)
    return online, {"best_val_TR": float(best_val_tr)}


def evaluate_dqn(net: _QNet, markets: list[DayMarket], cfg: Config) -> dict:
    net.eval()
    daily, fills = [], []
    with torch.no_grad():
        for m in markets:
            env = TradingEnv(m, cfg, hindsight=False)
            env.reset()
            while True:
                s = _state(m, env.t, env.pos, env.cash)
                a = int(net(torch.as_tensor(s)).argmax().item())
                if env.step(ACTION_MAP[a]).done:
                    break
            daily.append(env.net_value() - 1.0)
            fills.append(env.n_fills)
    out = financial_metrics(np.asarray(daily))
    out["daily_returns"] = [float(x) for x in daily]
    out["avg_fills_per_day"] = float(np.mean(fills))
    return out
