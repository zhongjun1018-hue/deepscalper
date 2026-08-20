"""分支 Q 智能体：ε-greedy 决策、定长区间折扣的 TD 更新、目标网络。

决策区间定长（decision_interval_min），TD 折扣恒为 gamma^decision_interval_min
（design 4.2）。数量档收缩为单档时网络只有半宽分支，动作元组的数量档恒为 0。
"""

from __future__ import annotations

import numpy as np
import torch

from .buffer import PrioritizedReplay, Transition
from .config import Config
from .env import DayMarket, Observation
from .model import BranchQNetwork, resolve_device, to_batch


def _branching_next_value(
    online_q: list[torch.Tensor],
    target_q: list[torch.Tensor],
    fixed_gears: tuple[int | None, ...],
    inactive_gears: tuple[int, ...],
    flatten_allowed: torch.Tensor,
) -> torch.Tensor:
    """以在线网络选动作、目标网络估值，并聚合有效的动作分支。

    平仓档只在下一状态净持仓非零时可选，动作选择与行为策略遵循同一掩码
    （design 5.1）。数量分支存在时（多档配置），半宽分支选择网格不触发档
    （平仓 / 关闭）则数量分支对执行无意义，延续价值只取半宽分支（design 6.3）。
    """
    q0 = online_q[0].clone()
    q0[~flatten_allowed, 0] = -torch.inf
    online_q = [q0, *online_q[1:]]
    actions, values = [], []
    for fixed, oq, tq in zip(fixed_gears, online_q, target_q):
        if fixed is None:
            action = oq.argmax(-1, keepdim=True)
        else:
            action = torch.full(
                (oq.shape[0], 1), fixed, dtype=torch.long, device=oq.device
            )
        actions.append(action.squeeze(1))
        values.append(tq.gather(1, action).squeeze(1))
    if len(values) == 1:
        return values[0]
    branch_values = torch.stack(values)
    joint = branch_values.mean(0)
    inactive = torch.isin(
        actions[0], torch.as_tensor(list(inactive_gears), device=actions[0].device)
    )
    return torch.where(inactive, branch_values[0], joint)


class BranchQAgent:
    """fixed_gears 给定某分支的档位索引时该分支不再探索（消融用），其余分支照常学习。"""

    def __init__(
        self,
        cfg: Config,
        seed: int,
        fixed_gears: tuple[int | None, int | None] = (None, None),
    ):
        self.cfg = cfg
        self.fixed_gears = fixed_gears
        self.inactive_gears = cfg.inactive_gears
        self.device = resolve_device(cfg)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = BranchQNetwork(cfg).to(self.device)
        self.target = BranchQNetwork(cfg).to(self.device).eval()
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = PrioritizedReplay(cfg.buffer_capacity, cfg.per_alpha)
        self.updates = 0

    # ---- 决策 ----
    def _apply_fixed(self, gears: list[int]) -> tuple[int, int]:
        for i, fixed in enumerate(self.fixed_gears[: len(gears)]):
            if fixed is not None:
                gears[i] = fixed
        return gears[0], gears[1] if len(gears) > 1 else 0

    def act(self, obs: Observation, epsilon: float) -> tuple[int, int]:
        if self.rng.random() < epsilon:
            # 平仓档（半宽档 0）只在净持仓非零时可选，探索与贪心遵循同一掩码
            lo = 0 if obs.flatten_allowed else 1
            gears = [int(self.rng.integers(lo, self.cfg.n_width))]
            if self.cfg.n_size > 1:
                gears.append(int(self.rng.integers(self.cfg.n_size)))
            return self._apply_fixed(gears)
        return self.greedy(obs)

    def greedy(self, obs: Observation) -> tuple[int, int]:
        self.online.eval()
        with torch.no_grad():
            q = self.online(*to_batch([obs], self.device))
        if not obs.flatten_allowed:
            q[0][:, 0] = -torch.inf
        return self._apply_fixed([int(b.argmax(-1).item()) for b in q])

    # ---- 学习 ----
    def push(self, tr: Transition) -> None:
        self.buffer.push(tr)

    def update(self, markets: list[DayMarket], beta: float) -> float | None:
        """更新一次，返回 Q 损失；样本不足以成批时返回 None。"""
        cfg = self.cfg
        if len(self.buffer) < cfg.batch_size:
            return None
        batch, idx, weights = self.buffer.sample(cfg.batch_size, beta, self.rng)

        obs = [markets[tr.day_id].observe(tr.t, tr.priv_hist) for tr in batch]
        next_obs = [
            None if tr.done else markets[tr.day_id].observe(tr.next_t, tr.next_priv_hist)
            for tr in batch
        ]

        self.online.train()
        q = self.online(*to_batch(obs, self.device))
        actions = torch.as_tensor([tr.action for tr in batch], device=self.device)
        q_sel = [qb.gather(1, actions[:, i, None]).squeeze(1) for i, qb in enumerate(q)]

        with torch.no_grad():
            rewards = torch.as_tensor(
                [tr.reward for tr in batch], dtype=torch.float32, device=self.device
            )
            next_value = torch.zeros(len(batch), device=self.device)
            keep = [i for i, o in enumerate(next_obs) if o is not None]
            if keep:
                kept_obs = [next_obs[i] for i in keep]
                next_batch = to_batch(kept_obs, self.device)
                online_q = self.online(*next_batch)
                target_q = self.target(*next_batch)
                index = torch.as_tensor(keep, device=self.device)
                flatten_allowed = torch.as_tensor(
                    [o.flatten_allowed for o in kept_obs], device=self.device
                )
                next_value[index] = _branching_next_value(
                    online_q, target_q, self.fixed_gears, self.inactive_gears,
                    flatten_allowed
                )
            # 定长区间：TD 折扣为常数 gamma^decision_interval_min（design 4.2）
            target = rewards + cfg.td_discount * next_value

        w = torch.as_tensor(weights, device=self.device)
        td = [target - qs for qs in q_sel]
        if len(td) == 1:
            sample_loss = td[0] ** 2
            priorities = td[0].abs()
        else:
            # 半宽分支恒有效；数量分支仅在网格可触发（非平仓 / 关闭档）时参与更新
            active = ~torch.isin(
                actions[:, 0],
                torch.as_tensor(list(self.inactive_gears), device=self.device),
            )
            masks = [torch.ones_like(active), active]
            n_active = 1.0 + active.float()
            sample_loss = sum(mask * error**2 for mask, error in zip(masks, td)) / n_active
            priorities = sum(mask * error.abs() for mask, error in zip(masks, td)) / n_active
        q_loss = (w * sample_loss).mean()

        self.optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self.buffer.update_priorities(idx, priorities.detach().cpu().numpy())

        self.updates += 1
        if self.updates % cfg.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(q_loss.item())
