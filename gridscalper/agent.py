"""BDQ 智能体：ε-greedy 决策、SMDP 折扣的 TD 更新、目标网络、波动率辅助损失。

总损失 L = L_q + η·L_vol；两个动作分支共享 BDQ 的联合 TD 目标。决策间隔 τ
由市场决定，故延续价值的折扣为 gamma^τ（gamma 为每 tick 口径，见 design 4.2）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import PrioritizedReplay, Transition
from .config import Config
from .env import DayMarket, Observation
from .model import BDQNetwork, resolve_device, to_batch


def _branching_next_value(
    online_q: list[torch.Tensor],
    target_q: list[torch.Tensor],
    fixed_gears: tuple[int | None, ...],
    inactive_gears: tuple[int, ...],
) -> torch.Tensor:
    """以在线网络选动作、目标网络估值，并聚合有效的 BDQ 分支。

    半宽分支（分支 0）选择网格不触发档（平仓 / 关闭）时，数量分支对执行无意义，
    延续价值只取半宽分支（design 6.3）。
    """
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
    branch_values = torch.stack(values)
    joint = branch_values.mean(0)
    inactive = torch.isin(
        actions[0], torch.as_tensor(list(inactive_gears), device=actions[0].device)
    )
    return torch.where(inactive, branch_values[0], joint)


class BDQAgent:
    """fixed_gears 给定某分支的档位索引时该分支不再探索（消融用），其余分支照常学习。"""

    def __init__(
        self,
        cfg: Config,
        seed: int,
        aux_task: bool = True,
        fixed_gears: tuple[int | None, int | None] = (None, None),
    ):
        self.cfg = cfg
        self.aux_task = aux_task
        self.fixed_gears = fixed_gears
        self.branch_sizes = (cfg.n_width, cfg.n_size)
        self.inactive_gears = cfg.inactive_gears
        self.device = resolve_device(cfg)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = BDQNetwork(cfg).to(self.device)
        self.target = BDQNetwork(cfg).to(self.device).eval()
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = PrioritizedReplay(cfg.buffer_capacity, cfg.per_alpha)
        self.updates = 0

    # ---- 决策 ----
    def _apply_fixed(self, gears: list[int]) -> tuple[int, int]:
        for i, fixed in enumerate(self.fixed_gears):
            if fixed is not None:
                gears[i] = fixed
        return gears[0], gears[1]

    def act(self, obs: Observation, epsilon: float) -> tuple[int, int]:
        if self.rng.random() < epsilon:
            return self._apply_fixed([int(self.rng.integers(n)) for n in self.branch_sizes])
        return self.greedy(obs)

    def greedy(self, obs: Observation) -> tuple[int, int]:
        self.online.eval()
        with torch.no_grad():
            q, _ = self.online(*to_batch([obs], self.device))
        return self._apply_fixed([int(b.argmax(-1).item()) for b in q])

    # ---- 学习 ----
    def push(self, tr: Transition) -> None:
        self.buffer.push(tr)

    def update(self, markets: list[DayMarket], beta: float) -> tuple[float, float] | None:
        """更新一次，返回 (Q 损失, 波动率辅助损失)；样本不足以成批时返回 None。"""
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
        q, vol_pred = self.online(*to_batch(obs, self.device))
        actions = torch.as_tensor([tr.action for tr in batch], device=self.device)
        q_sel = [qb.gather(1, actions[:, i, None]).squeeze(1) for i, qb in enumerate(q)]

        with torch.no_grad():
            rewards = torch.as_tensor(
                [tr.reward for tr in batch], dtype=torch.float32, device=self.device
            )
            # SMDP 折扣：区间跨 τ 个 tick，权重为每 tick 折扣的 τ 次幂
            discount = torch.as_tensor(
                [cfg.gamma**tr.tau for tr in batch], dtype=torch.float32, device=self.device
            )
            next_value = torch.zeros(len(batch), device=self.device)
            keep = [i for i, o in enumerate(next_obs) if o is not None]
            if keep:
                next_batch = to_batch([next_obs[i] for i in keep], self.device)
                online_q, _ = self.online(*next_batch)
                target_q, _ = self.target(*next_batch)
                index = torch.as_tensor(keep, device=self.device)
                next_value[index] = _branching_next_value(
                    online_q, target_q, self.fixed_gears, self.inactive_gears
                )
            target = rewards + discount * next_value

        w = torch.as_tensor(weights, device=self.device)
        td = [target - qs for qs in q_sel]
        # 半宽分支恒有效；数量分支仅在网格可触发（非平仓 / 关闭档）时参与更新
        active = ~torch.isin(
            actions[:, 0],
            torch.as_tensor(list(self.inactive_gears), device=self.device),
        )
        masks = [torch.ones_like(active), active]
        n_active = 1.0 + active.float()
        sample_loss = sum(mask * error**2 for mask, error in zip(masks, td)) / n_active
        q_loss = (w * sample_loss).mean()

        vol_loss = torch.zeros((), device=self.device)
        if self.aux_task:
            vol_target = torch.as_tensor(
                [tr.vol_label for tr in batch], dtype=torch.float32, device=self.device
            )
            vol_loss = F.mse_loss(vol_pred, vol_target)
        loss = q_loss + cfg.vol_loss_weight * vol_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.optimizer.step()

        priorities = (
            sum(mask * error.abs() for mask, error in zip(masks, td)) / n_active
        ).detach().cpu().numpy()
        self.buffer.update_priorities(idx, priorities)

        self.updates += 1
        if self.updates % cfg.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(q_loss.item()), float(vol_loss.item())
