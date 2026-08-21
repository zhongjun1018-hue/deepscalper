"""逐日 K 线与固定网格半宽：特征构建、tick ATR 与回测共用的宽度口径。

网格间距为 atr_mult×ATR（ATR 取前 atr_window 个完整交易日 TR 均值），参数
默认值统一定义在缓存规格 WindowSpec（data_provider/windows.py），此处不设缺省。
"""

import numpy as np
import pandas as pd


def daily_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """逐日 high / low / pre_close（来自 HighPx / LowPx / PreClosePx），剔除无效日。"""
    grouped = frame.groupby("MDDate", sort=True)
    bars = pd.DataFrame({
        "high": grouped["HighPx"].max(),
        "low": grouped["LowPx"].apply(lambda values: values[values > 0].min()),
        "preclose": grouped["PreClosePx"].last(),
    })
    return bars[(bars["high"] > 0) & (bars["low"] > 0) & (bars["preclose"] > 0)]


def true_range(bars: pd.DataFrame) -> pd.Series:
    """逐日真实波幅 TR = max(高−低, |高−前收|, |低−前收|)。"""
    return pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - bars["preclose"]).abs(),
        (bars["low"] - bars["preclose"]).abs(),
    ], axis=1).max(axis=1)


def grid_width(bars: pd.DataFrame, atr_mult: float, atr_window: int,
               min_width_ratio: float) -> pd.Series:
    """逐日固定半宽 W = max(min_width_ratio×前收, atr_mult×ATR)，按 MDDate 索引。

    ATR 为前 atr_window 个完整交易日 TR 的均值（shift(1)，无前视）；
    历史不足的日期不出现在结果中（调用方按缺失处理）。
    """
    atr = true_range(bars).rolling(atr_window).mean().shift(1)
    width = np.maximum(min_width_ratio * bars["preclose"], atr_mult * atr)
    return pd.Series(width, index=bars.index).dropna()
