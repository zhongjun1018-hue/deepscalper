"""策略评估指标：财务指标（TR/SR/CR/SoR，RL 实验用）与统一回测的逐日汇总。

财务指标以逐日超额收益序列为基础（日频不年化，design 7.3）；summarize 把统一回测的
逐日记录汇总为以去量纲网格收益 g 为主的同一套指标（docs/grid_profit.md §七 的
费用后口径，design 8.5），各模式可直接对比。
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


def _nan_moments(values: np.ndarray) -> tuple[float, float]:
    """有值日的均值与总体标准差；全 NaN 记 (NaN, NaN)。"""
    if not np.isfinite(values).any():
        return float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanstd(values))


def _nonzero_mean(values: np.ndarray) -> float:
    """总和除以该指标非零日的日数；全为零记 NaN。"""
    count = int(np.count_nonzero(values))
    return float(values.sum() / count) if count else float("nan")


def _traded_mean(values: np.ndarray, trades: np.ndarray) -> float:
    """有成交日的均值（零交易日剔除出分母）；全无成交记 NaN。"""
    traded = trades > 0
    return float(values[traded].mean()) if traded.any() else float("nan")


def summarize(days: list[dict]) -> dict[str, float]:
    """逐日记录的统一汇总：费用后网格收益 g + 日均闭环率 / 成交次数 / 日终敞口 /
    满闭环率 / 宽幅。

    days 元素为 {g, n_buys, n_sells, closure_rate, width_rel}。g 以当日基准
    格距 W_d 归一（design 8.5）；width_rel 为当日时间加权生效半宽 / 当日开盘价，
    无网格触发时间的日记 NaN。g 与宽幅的矩按有值日计。

    闭环率与买卖笔数取「总和 / 该指标非零日的日数」（design 8.5）：零值日既不
    贡献分子也不摊薄分母，全为零时记 NaN。日终敞口 |N_b − N_s| 与满闭环率
    （闭环率为 1 的日占比）按有成交日平均，零交易日剔除出分母。
    """
    buys = np.array([d["n_buys"] for d in days], dtype=np.float64)
    sells = np.array([d["n_sells"] for d in days], dtype=np.float64)
    closure = np.array([d["closure_rate"] for d in days], dtype=np.float64)
    trades = buys + sells
    mean_g, std_g = _nan_moments(np.array([d["g"] for d in days], dtype=np.float64))
    return {
        "n_days": len(days),
        "mean_g": mean_g,
        "std_g": std_g,
        "mean_closure_rate": _nonzero_mean(closure),
        "mean_trades": _nonzero_mean(trades),
        "mean_buys": _nonzero_mean(buys),
        "mean_sells": _nonzero_mean(sells),
        "mean_abs_exposure": _traded_mean(np.abs(buys - sells), trades),
        "full_closure_rate": _traded_mean((closure == 1.0).astype(np.float64), trades),
        "mean_width_rel": _nan_moments(
            np.array([d["width_rel"] for d in days], dtype=np.float64))[0],
    }
