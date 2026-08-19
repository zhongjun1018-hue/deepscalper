"""门控信号：预测算法控制网格策略启停的决策机制。

残差趋势门控：resid_abs_q90/w < 1.3 且 abs_slope/w > 0.9 时排除，
其中 w = 当日半宽/价格（判定 tick 的中间价）。

取数 scheme：none（常开）/ oracle（统一缓存 targets 真值）/ prediction（同一缓存的
preds 预测值）。信号逐 tick 判定：tick t 的信号只用截至 t 的回看窗口，命中即表示该
tick 不开新网。

门控由 strategy/engine 执行——只在净持仓为 0 时读取当拍信号：空仓期间每
stride_ticks 个 tick 复判一次，敞口归零即刻重判；预测算法本身不下单，网格几何与撮合
由 strategy 承担。
"""

from __future__ import annotations

import numpy as np

from data_provider.windows import TARGET_NAMES

GATES = ["residual"]
SCHEMES = ["none", "oracle", "prediction"]
DEFAULT_RESIDUAL_RATIO_THRESHOLD = 1.3
DEFAULT_SLOPE_RATIO_THRESHOLD = 0.9

_TARGET_INDEX = {name: index for index, name in enumerate(TARGET_NAMES)}
# 门控信号以当日相对半宽 w 为单位
RATIO_TARGETS = {"resid_abs_q90", "abs_slope"}


def gate_rules(residual_ratio_threshold=DEFAULT_RESIDUAL_RATIO_THRESHOLD,
               slope_ratio_threshold=DEFAULT_SLOPE_RATIO_THRESHOLD) -> dict:
    """门控的判定规则：{门控: ((信号名, 比较符, 阈值), ...)}，全部条件成立才排除。"""
    return {
        "residual": (("resid_abs_q90", "<", residual_ratio_threshold),
                     ("abs_slope", ">", slope_ratio_threshold)),
    }


GATE_RULES = gate_rules()


def build_gate_masks(sources, dates, widths, anchors, gates=GATES, rules=GATE_RULES):
    """逐 (gate, scheme, day) 的逐 tick 布尔掩码：True 表示该 tick 的信号排除开新网。

    sources: {"oracle": (D,T,5) 缓存 targets, "prediction": (D,T,5) 缓存 preds}；
    dates: (D,) 日期字符串；widths: (D,) 当日半宽；anchors: (D,T) 各 tick 的有效 mid
    （w = widths/anchors）。oracle 与 prediction 任一信号非有限的 tick 两侧都不判定。

    返回 (masks, gated)：masks 键为 (gate, scheme, day)、值为 (T,) bool（无命中的日
    不建键）；gated 为 {gate: {"total", "oracle", "prediction"}} 触发计数。
    """
    widths = np.asarray(widths, dtype=np.float64)
    anchors = np.asarray(anchors, dtype=np.float64)
    valid = np.isfinite(widths) & (widths > 0)
    relative = (np.where(valid, widths, np.nan)[:, None]
                / np.where(anchors > 0, anchors, np.nan))

    masks, gated = {}, {}
    for gate in gates:
        rule = rules[gate]
        hits, judgeable = {}, np.ones(relative.shape, dtype=bool)
        for scheme in SCHEMES[1:]:
            signals = [
                sources[scheme][..., _TARGET_INDEX[name]] / relative
                if name in RATIO_TARGETS else sources[scheme][..., _TARGET_INDEX[name]]
                for name, _, _ in rule
            ]
            judgeable &= np.logical_and.reduce([np.isfinite(v) for v in signals])
            hits[scheme] = np.logical_and.reduce(
                [v > threshold if operator == ">" else v < threshold
                 for v, (_, operator, threshold) in zip(signals, rule)])
        gated[gate] = {"total": int(judgeable.sum())}
        for scheme, hit in hits.items():
            mask = hit & judgeable
            gated[gate][scheme] = int(mask.sum())
            for d, day in enumerate(dates):
                if mask[d].any():
                    masks[(gate, scheme, str(day))] = mask[d]
    return masks, gated
