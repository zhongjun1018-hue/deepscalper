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
from .features import MACRO_DIM, MICRO_DIM, PRIVATE_DIM
from .metrics import financial_metrics
from .model import resolve_device

STATE_DIM = MACRO_DIM + MICRO_DIM + PRIVATE_DIM


def _action_map(cfg: Config) -> dict[int, tuple[int, int]]:
    """3 个粗动作 → BDQ 动作网格索引 (穿价档深, 数量)。

    price_levels / quantities 均按升序排列，故末档为最深穿价与最大多头数量、
    首个数量档为最大空头数量；买卖均取最深穿价档以模拟市价单，持有取零数量。
    """
    deepest = cfg.n_price - 1
    return {
        0: (deepest, 0),                              # 做空
        1: (deepest, cfg.quantities.index(0)),        # 持有
        2: (deepest, cfg.n_quantity - 1),             # 做多
    }


def _push(replay: list, pos: int, capacity: int, item: tuple) -> int:
    """环形写入回放缓冲区，返回下一个写入位置（容量满后覆盖最旧样本）。"""
    if len(replay) < capacity:
        replay.append(item)
    else:
        replay[pos] = item
    return (pos + 1) % capacity


class _QNet(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x):
        return self.net(x)


def _state(env: TradingEnv) -> np.ndarray:
    """DQN 的平铺状态：宏观向量 + 末帧微观特征 + 当期私有状态。"""
    obs = env.observation()
    return np.concatenate([obs.macro, obs.micro_lob[-1], obs.private[-1]]).astype(np.float32)


def train_dqn(
    cfg: Config,
    markets: list[DayMarket],
    val_markets: list[DayMarket],
    seed: int,
    log_prefix: str = "[DQN]",
) -> tuple[_QNet, dict]:
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(seed)
    device = resolve_device(cfg)
    rng = np.random.default_rng(seed)
    action_map = _action_map(cfg)
    online, target = _QNet(cfg.hidden_size).to(device), _QNet(cfg.hidden_size).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    replay: list[tuple] = []
    replay_pos = 0
    updates = 0

    best_val_tr, best_state = -np.inf, None
    for epoch in range(1, cfg.epochs + 1):
        online.train()
        for day_id, m in enumerate(markets):
            env = TradingEnv(m, cfg, hindsight=False)
            eps = cfg.epsilon_at(epoch, day_id / max(1, len(markets)))
            s = _state(env)
            while True:
                if rng.random() < eps:
                    a = int(rng.integers(3))
                else:
                    with torch.no_grad():
                        a = int(online(torch.as_tensor(s, device=device)).argmax().item())
                res = env.step(action_map[a])
                if res.done:
                    replay_pos = _push(replay, replay_pos, cfg.buffer_capacity,
                                       (s, a, res.reward, None))
                    break
                s2 = _state(env)
                replay_pos = _push(replay, replay_pos, cfg.buffer_capacity,
                                   (s, a, res.reward, s2))
                if len(replay) >= cfg.batch_size and env.step_idx % cfg.update_every == 0:
                    idx = rng.choice(len(replay), cfg.batch_size, replace=True)
                    batch = [replay[i] for i in idx]
                    S = torch.as_tensor(np.stack([b[0] for b in batch]), device=device)
                    A = torch.as_tensor([b[1] for b in batch], device=device)
                    R = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=device)
                    nonterm = [b[3] for b in batch]
                    with torch.no_grad():
                        nq = torch.zeros(len(batch), device=device)
                        keep = [i for i, x in enumerate(nonterm) if x is not None]
                        if keep:
                            ns = torch.as_tensor(np.stack([nonterm[i] for i in keep]), device=device)
                            nq[keep] = target(ns).max(-1).values
                        y = R + cfg.gamma * nq
                    loss = nn.functional.mse_loss(online(S).gather(1, A[:, None]).squeeze(1), y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    updates += 1
                    if updates % cfg.target_sync == 0:
                        target.load_state_dict(online.state_dict())
                s = s2

        val = evaluate_dqn(online, val_markets, cfg)
        print(f"{log_prefix} epoch {epoch}/{cfg.epochs} val_TR={val['TR']:.4f} val_SR={val['SR']:.3f}", flush=True)
        if val["TR"] > best_val_tr:
            best_val_tr, best_state = val["TR"], {k: v.clone() for k, v in online.state_dict().items()}

    if best_state is not None:
        online.load_state_dict(best_state)
    return online, {"best_val_TR": float(best_val_tr)}


def evaluate_dqn(net: _QNet, markets: list[DayMarket], cfg: Config) -> dict:
    net.eval()
    device = next(net.parameters()).device
    action_map = _action_map(cfg)
    daily, fills = [], []
    with torch.no_grad():
        for m in markets:
            env = TradingEnv(m, cfg, hindsight=False)
            while True:
                s = _state(env)
                a = int(net(torch.as_tensor(s, device=device)).argmax().item())
                if env.step(action_map[a]).done:
                    break
            daily.append(env.net_value() - 1.0)
            fills.append(env.n_fills)
    out = financial_metrics(np.asarray(daily))
    out["daily_returns"] = [float(x) for x in daily]
    out["avg_fills_per_day"] = float(np.mean(fills))
    return out
