"""BDQ 智能体：ε-greedy 决策、一步 TD 更新、目标网络、波动率辅助损失。

总损失 L = L_q + η·L_vol（论文 4.4）；L_q 为价格 / 数量两分支 TD 误差均值。
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import PrioritizedReplay, Transition
from .config import Config
from .env import DayMarket, Observation
from .model import BDQNetwork, to_batch


class BDQAgent:
    def __init__(self, cfg: Config, seed: int, aux_task: bool = True):
        self.cfg = cfg
        self.aux_task = aux_task
        self.device = torch.device("cpu")
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = BDQNetwork(cfg)
        self.target = copy.deepcopy(self.online).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = PrioritizedReplay(cfg.buffer_capacity, cfg.per_alpha)
        self.updates = 0

    # ---- 决策 ----
    def act(self, obs: Observation, epsilon: float) -> tuple[int, int]:
        if self.rng.random() < epsilon:
            return (
                int(self.rng.integers(self.cfg.n_price)),
                int(self.rng.integers(self.cfg.n_quantity)),
            )
        return self.greedy(obs)

    def greedy(self, obs: Observation) -> tuple[int, int]:
        self.online.eval()
        with torch.no_grad():
            q_p, q_q, _ = self.online(*to_batch([obs], self.device))
        return int(q_p.argmax(-1).item()), int(q_q.argmax(-1).item())

    # ---- 学习 ----
    def push(self, tr: Transition) -> None:
        self.buffer.push(tr)

    def _obs_of(self, markets: list[DayMarket], day_id: int, t_idx: int,
                pos: float, cash: float) -> Observation:
        m = markets[day_id]
        return m.observe(m.decision_points[t_idx], pos, cash, m.cash0)

    def update(self, markets: list[DayMarket], beta: float) -> float | None:
        cfg = self.cfg
        if len(self.buffer) < cfg.batch_size:
            return None
        batch, idx, weights = self.buffer.sample(cfg.batch_size, beta, self.rng)

        obs = [self._obs_of(markets, tr.day_id, tr.t, tr.pos, tr.cash) for tr in batch]
        next_obs = [
            self._obs_of(markets, tr.day_id, tr.next_t, tr.next_pos, tr.next_cash)
            if not tr.done else None
            for tr in batch
        ]

        self.online.train()
        q_p, q_q, vol_pred = self.online(*to_batch(obs, self.device))
        a_p = torch.as_tensor([tr.action_p for tr in batch])
        a_q = torch.as_tensor([tr.action_q for tr in batch])
        q_p_sel = q_p.gather(1, a_p[:, None]).squeeze(1)
        q_q_sel = q_q.gather(1, a_q[:, None]).squeeze(1)

        with torch.no_grad():
            rewards = torch.as_tensor([tr.reward for tr in batch], dtype=torch.float32)
            dones = torch.as_tensor([tr.done for tr in batch], dtype=torch.float32)
            non_terminal = [o for o in next_obs if o is not None]
            next_q_p = torch.zeros(len(batch))
            next_q_q = torch.zeros(len(batch))
            if non_terminal:
                nq_p, nq_q, _ = self.target(*to_batch(non_terminal, self.device))
                it = iter(zip(nq_p.max(-1).values, nq_q.max(-1).values))
                for i, o in enumerate(next_obs):
                    if o is not None:
                        vp, vq = next(it)
                        next_q_p[i], next_q_q[i] = vp, vq
            y_p = rewards + cfg.gamma * (1 - dones) * next_q_p
            y_q = rewards + cfg.gamma * (1 - dones) * next_q_q

        w = torch.as_tensor(weights)
        td_p = y_p - q_p_sel
        td_q = y_q - q_q_sel
        loss_q = ((w * (td_p**2 + td_q**2)) / 2).mean()

        loss = loss_q
        if self.aux_task:
            vol_target = torch.as_tensor([tr.vol_label for tr in batch], dtype=torch.float32)
            loss = loss + cfg.vol_loss_weight * F.mse_loss(vol_pred, vol_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        td = ((td_p.abs() + td_q.abs()) / 2).detach().numpy()
        self.buffer.update_priorities(idx, td)

        self.updates += 1
        if self.updates % cfg.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
