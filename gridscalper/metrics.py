"""评估指标：TR / SR / CR / SoR，以及网格策略的补充指标（design 7.4）。

以测试期逐日超额净值序列为基础：日收益 r_d = 当日净值 − 1（已扣除底仓 beta），
TR 为复利累计收益，SR = E[r]/σ[r]，CR = E[r]/MDD，SoR = E[r]/下行波动。
"""

from __future__ import annotations

import numpy as np


def financial_metrics(daily_returns: np.ndarray) -> dict[str, float]:
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


def aggregate_diagnostics(logs: list[dict]) -> dict:
    """将 TradingEnv.episode_log 的逐日指标按日平均（分布类字段逐档平均）。"""
    if not logs:
        return {}
    out = {}
    for key, sample in logs[0].items():
        values = [log[key] for log in logs]
        out[key] = (np.mean(values, axis=0).tolist() if isinstance(sample, list)
                    else float(np.mean(values)))
    return out
