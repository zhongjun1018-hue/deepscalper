"""策略评估指标：财务指标（TR/SR/CR/SoR）与网格形态指标（Score、成交笔数、每笔利润）。

财务指标以逐日超额净值序列为基础（日频不年化）；网格形态指标以 engine.run_day
的逐日回放结果为基础，字段名与特征/回测两侧的消费方保持一致。
"""

from __future__ import annotations

import numpy as np


def financial_metrics(daily_returns: np.ndarray) -> dict[str, float]:
    """TR / SR / CR / SoR：日收益 r_d = 当日净值 − 1，日频不年化；分母为 0 记 0。

    MDD 的峰值序列含初始净值 1，首日回撤不漏计。
    """
    r = np.asarray(daily_returns, dtype=np.float64)
    if r.size == 0:
        return {"TR": 0.0, "SR": 0.0, "CR": 0.0, "SoR": 0.0}
    cum = np.cumprod(1.0 + r)
    tr = cum[-1] - 1.0
    mean = r.mean()
    std = r.std()
    sr = mean / std if std > 1e-12 else 0.0
    nav = np.concatenate([[1.0], cum])   # 峰值序列含初始净值，首日回撤不漏计
    peak = np.maximum.accumulate(nav)
    mdd = float(((peak - nav) / peak).max())
    cr = mean / mdd if mdd > 1e-12 else 0.0
    dd = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    sor = mean / dd if dd > 1e-12 else 0.0
    return {"TR": float(tr), "SR": float(sr), "CR": float(cr), "SoR": float(sor)}


def closure_rate(n_buys: int, n_sells: int) -> float:
    """当日买卖笔数的配对比例 2·min(Nb, Ns)/(Nb + Ns)：1 为完全配对，0 为纯单边（零成交记 0）。"""
    total = n_buys + n_sells
    return 2.0 * min(n_buys, n_sells) / total if total else 0.0


def day_frame(records: list[dict]) -> dict[str, np.ndarray]:
    """逐日网格指标表：buys / sells / grid_profit / grid_profit_lower 及派生列。"""
    keys = ["buys", "sells", "grid_profit", "grid_profit_lower"]
    fields = {key: np.array([r[key] for r in records]) for key in keys}
    fields["trades"] = fields["buys"] + fields["sells"]
    fields["rounds"] = np.minimum(fields["buys"], fields["sells"])
    fields["exposure"] = fields["buys"] - fields["sells"]
    return fields


def _moments(values, weights=None):
    if len(values) == 0:
        return float("nan"), float("nan")
    if weights is None:
        return float(values.mean()), float(values.std(ddof=0))
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    return float(mean), float(np.sqrt(variance))


def summarize(days: dict[str, np.ndarray]) -> dict[str, float]:
    """形态汇总：Score = 2·min(Nb, Ns)/N 仅在成交日取矩，笔数类统计覆盖全部交易日。"""
    trades = days["trades"]
    rounds = days["rounds"]
    filled = trades > 0
    # 失衡度 |N_b − N_s| / N_d 是该 Score 的补，故只报告其一
    score = 2 * rounds[filled] / trades[filled]
    weighted = _moments(score, trades[filled])
    equal = _moments(score)
    closed = days["exposure"][filled] == 0
    grid_profit = days["grid_profit"]
    grid_profit_lower = days["grid_profit_lower"]
    profit_per_trade = grid_profit[filled] / trades[filled]
    return {
        "weighted_score_mean": weighted[0],
        "weighted_score_std": weighted[1],
        "equal_score_mean": equal[0],
        "equal_score_std": equal[1],
        "n_days": int(len(trades)),
        "n_scored": int(filled.sum()),
        "closed_day_share": float(closed.mean()) if len(closed) else float("nan"),
        "mean_rounds": float(rounds.mean()) if len(trades) else float("nan"),
        "mean_buys": float(days["buys"].mean()) if len(trades) else float("nan"),
        "mean_sells": float(days["sells"].mean()) if len(trades) else float("nan"),
        "mean_grid_profit": float(grid_profit.mean()) if len(trades) else float("nan"),
        "std_grid_profit": float(grid_profit.std(ddof=0)) if len(trades) else float("nan"),
        "mean_grid_profit_lower": float(grid_profit_lower.mean()) if len(trades) else float("nan"),
        "std_grid_profit_lower": float(grid_profit_lower.std(ddof=0)) if len(trades) else float("nan"),
        "mean_profit_per_trade": (float(profit_per_trade.mean())
                                  if len(profit_per_trade) else float("nan")),
        "std_profit_per_trade": (float(profit_per_trade.std(ddof=0))
                                 if len(profit_per_trade) else float("nan")),
    }
