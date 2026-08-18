"""基线方法：常开网格、固定半宽的单维扫描与买入持有底仓（design 7.2）。

三者与 RL 共用同一 TradingEnv，撮合、成本、账户与日终平回底仓的规则完全一致，
差别只在参数恒定；扫描曲面是无惩罚（风险中性）目标的经验参照，
其逐格点的敞口负载用于与 RL 做同风险比较。
"""

from __future__ import annotations

import numpy as np

from .env import DayMarket, GridParams
from .metrics import financial_metrics
from .train import evaluate

OPEN_WIDTH = 0.1                                # 常开参照基线的半宽（× ATR3）
SCAN_WIDTHS = (0.05, 0.1, 0.15, 0.2, 0.25)      # × ATR3


def run_fixed_grid(markets: list[DayMarket], half_width: float, size: int = 1) -> dict:
    """固定半宽的对称网格，全天不平仓、不做任何控制。

    half_width 以 ATR3 为单位，不必落在动作梯子上——规则层与档位表无关。
    """
    params = GridParams(half_width, size=size)
    return evaluate(markets, lambda obs: params)


def run_open_grid(markets: list[DayMarket]) -> dict:
    """常开网格：成功判定的参照基线（design 7.2）。"""
    return {**run_fixed_grid(markets, OPEN_WIDTH), "half_width": OPEN_WIDTH}


def run_grid_scan(val_markets: list[DayMarket], test_markets: list[DayMarket]) -> dict:
    """固定半宽网格的单维扫描：格点在验证集上按 SR 选优，报告该格点的测试集指标。

    选优协议与 RL 一致，避免在测试集上挑参数；`points` 保留全部格点的测试集结果，
    其 `inventory_load` 为时间加权 $(I/B)^2$，即该格点承担的风险（design 7.4）。
    """
    val_sr = [run_fixed_grid(val_markets, h)["SR"] for h in SCAN_WIDTHS]
    points = [{"half_width": h, **run_fixed_grid(test_markets, h)} for h in SCAN_WIDTHS]
    best = points[int(np.argmax(val_sr))]
    return {**{k: best[k] for k in ("TR", "SR", "CR", "SoR", "daily_returns",
                                    "daily_closure_rate", "diagnostics")},
            "best_point": {"half_width": best["half_width"]},
            "points": points}


def run_hold_base(markets: list[DayMarket]) -> dict:
    """买入持有底仓：报告被 7.1 从奖励中扣除的 beta，即底仓的逐日中间价收益。"""
    daily = np.asarray([m.mid[m.n - 1] / m.p0 - 1.0 for m in markets])
    return {**financial_metrics(daily), "daily_returns": daily.tolist()}
