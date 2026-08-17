"""评估指标：TR / SR / CR / SoR（论文 5.2）。

以测试期逐日净值序列为基础：日收益 r_d = 当日净值 − 1（每日平仓、净值归一），
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
    downside = r[r < 0]
    dd = downside.std() if downside.size > 1 else 0.0
    sor = mean / dd if dd > 1e-12 else 0.0
    return {"TR": float(tr), "SR": float(sr), "CR": float(cr), "SoR": float(sor)}
