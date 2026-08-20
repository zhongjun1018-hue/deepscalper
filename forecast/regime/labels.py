"""模式定义：残差趋势模式的事后标签（规则观测 + 粘性平滑）。

网格不利模式为「低残差、强趋势」——低震荡路径上网格被单边穿越。逐锚点规则
`resid_abs_q90/w < 残差阈值 ∧ abs_slope/w > 斜率阈值`（w 为当日相对半宽）给出
带噪观测；分钟锚点序列上以粘性转移（对角 sticky_stay）、对称混淆（emission_noise）
做前向-后向平滑，硬标签取平滑后验——状态连片，消除逐拍真值抖动。不可判定记 -1。

平滑用了整日序列（含未来），因此标签只作事后真值：训练识别器的目标与回测的
oracle 参照，不直接用于盘中决策。
"""

from __future__ import annotations

import numpy as np

from forecast.regime.config import RegimeConfig
from forecast.regime.data import RESID_IDX, SLOPE_IDX, SymbolBank


def rule_hits(bank: SymbolBank, cfg: RegimeConfig) -> np.ndarray:
    """(D,M) int8 逐锚点规则观测：1 = 命中（不利），0 = 未命中，-1 = 不可判定。"""
    with np.errstate(invalid="ignore"):
        hit = (bank.judgeable
               & (bank.realized[..., RESID_IDX] < cfg.residual_ratio_threshold)
               & (bank.realized[..., SLOPE_IDX] > cfg.slope_ratio_threshold))
    return np.where(bank.judgeable, hit.astype(np.int8), np.int8(-1))


def pattern_labels(bank: SymbolBank, cfg: RegimeConfig) -> np.ndarray:
    """(D,M) int8 事后模式标签：规则观测经粘性平滑（1 = 不利，-1 不可判定）。"""
    observed = rule_hits(bank, cfg)
    noise = cfg.emission_noise
    out = np.full(observed.shape, -1, dtype=np.int8)
    for i in range(len(observed)):
        obs = observed[i]
        if not (obs >= 0).any():
            continue
        emission = np.ones((len(obs), 2))   # 不可判定拍发射均匀，仅靠转移传播
        emission[obs == 1] = (noise, 1.0 - noise)
        emission[obs == 0] = (1.0 - noise, noise)
        posterior = _forward_backward(emission, cfg.sticky_stay)
        out[i] = (posterior > 0.5).astype(np.int8)
    out[~bank.judgeable] = -1
    return out


def _forward_backward(emission, stay: float) -> np.ndarray:
    """二状态粘性转移的标准前向-后向，返回状态 1（不利）的平滑后验 (S,)。"""
    n = len(emission)
    transition = np.array([[stay, 1.0 - stay], [1.0 - stay, stay]])
    forward = np.zeros((n, 2))
    backward = np.ones((n, 2))
    scale = np.zeros(n)
    forward[0] = 0.5 * emission[0]
    scale[0] = forward[0].sum()
    forward[0] /= scale[0]
    for t in range(1, n):
        forward[t] = (forward[t - 1] @ transition) * emission[t]
        scale[t] = forward[t].sum()
        forward[t] /= scale[t]
    for t in range(n - 2, -1, -1):
        backward[t] = transition @ (backward[t + 1] * emission[t + 1]) / scale[t + 1]
    posterior = forward * backward
    return posterior[:, 1] / posterior.sum(axis=1)
