"""交易日切分：全项目唯一的 train/val/test 划分（按时间排序的单次切分）。

确定性计算，直接按日期列表求值，不持久化、不做哈希缓存。
"""

from __future__ import annotations

from dataclasses import dataclass

SPLIT_RATIOS = (0.7, 0.1, 0.2)  # 训练 : 验证 : 测试


@dataclass(frozen=True)
class Split:
    """单次时序三段切分的结果（日期列表，互不重叠、覆盖全部输入日期）。"""

    train: list
    val: list
    test: list


def chronological_split(dates: list, ratios: tuple = SPLIT_RATIOS) -> Split:
    """按时间排序后单次切分：n_train = int(n×ratios[0])，n_val = int(n×ratios[1])，其余测试。

    任一段为空抛 ValueError。
    """
    ordered = sorted(dates)
    n = len(ordered)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    if min(n_train, n_val, n - n_train - n_val) <= 0:
        raise ValueError(f"{n} 个交易日按 {tuple(ratios)} 切分存在空段")
    return Split(
        train=ordered[:n_train],
        val=ordered[n_train:n_train + n_val],
        test=ordered[n_train + n_val:],
    )
