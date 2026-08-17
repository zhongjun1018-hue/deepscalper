"""优先经验回放（proportional PER）。

转移样本以轻量引用存储（交易日 id + 决策时刻 + 动作 + 私有状态历史），
观测在采样时由 DayMarket 重建，避免在缓冲区中冗余保存微观 / 宏观特征矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    day_id: int
    t: int                 # 决策点在 decision_points 中的索引
    action_p: int
    action_q: int
    reward: float             # 训练奖励（含 hindsight bonus，若启用）
    next_t: int               # -1 表示终止
    done: bool
    priv_hist: np.ndarray     # 决策时（动作前）的 (持仓, 现金) 历史
    next_priv_hist: np.ndarray
    vol_label: float


class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.data: list[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def push(self, tr: Transition) -> None:
        """新样本以当前最大优先级入队，保证至少被采样一次。"""
        prio = self.priorities[: len(self.data)].max() if self.data else 1.0
        if len(self.data) < self.capacity:
            self.data.append(tr)
        else:
            self.data[self.pos] = tr
        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float, rng: np.random.Generator):
        n = len(self.data)
        prob = self.priorities[:n] ** self.alpha
        prob /= prob.sum()
        idx = rng.choice(n, size=batch_size, replace=True, p=prob)
        weights = (n * prob[idx]) ** (-beta)
        weights /= weights.max()
        batch = [self.data[i] for i in idx]
        return batch, idx, weights.astype(np.float32)

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        self.priorities[idx] = np.abs(td_errors) + 1e-6
