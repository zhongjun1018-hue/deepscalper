"""数据加载：读取月度 tick parquet，过滤连续竞价时段，按交易日组织并预计算 ATR。"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

# A 股连续竞价时段：上午 09:30:00–11:30:00，下午 13:00:00–14:56:59。
# 上午含 11:30:00（该快照为上午收盘瞬时状态），下午不含 14:57:00（收盘集合竞价起点）。
MORNING_OPEN, MORNING_CLOSE = time(9, 30), time(11, 30)
AFTERNOON_OPEN, AFTERNOON_CLOSE = time(13, 0), time(14, 57)

PRICE_COLS = [f"{s}{i}Price" for s in ("Buy", "Sell") for i in range(1, 11)]
QTY_COLS = [f"{s}{i}OrderQty" for s in ("Buy", "Sell") for i in range(1, 11)]
ORDERS_COLS = [f"{s}{i}NumOrders" for s in ("Buy", "Sell") for i in range(1, 11)]

READ_COLS = [
    "MDDate", "MDTime", "LastPx", "PreClosePx", "HighPx", "LowPx",
    "TotalVolumeTrade", "NumTrades",
    "TotalBidQty", "TotalOfferQty", "NumBidOrders", "NumOfferOrders",
    "WeightedAvgBidPx", "WeightedAvgOfferPx",
    "WithdrawBuyAmount", "WithdrawSellAmount",
] + PRICE_COLS + QTY_COLS + ORDERS_COLS


@dataclass
class DayData:
    """单个交易日的连续竞价快照序列（已按时间排序、剔除无效行）。"""

    date: str
    frame: pd.DataFrame  # 索引 0..N-1，含 READ_COLS 全部字段
    pre_close: float     # 前收盘价
    atr: float           # 前 A 个完整交易日真实波幅的均值（价格计）；历史不足记 nan


def _read_symbol(symbol: str, data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, symbol, "*_tick.parquet")))
    if not files:
        raise FileNotFoundError(f"未找到 {symbol} 的 tick 数据：{data_dir}/{symbol}")
    return pd.concat((pd.read_parquet(f, columns=READ_COLS) for f in files), ignore_index=True)


def _true_range(frame: pd.DataFrame, pre_close: float) -> float:
    """当日真实波幅 TR = max{高−低, |高−前收|, |低−前收|}（design 2.1）。

    HighPx / LowPx 为快照内的日累计极值，取整日极值即为当日高低点。
    """
    if not pre_close > 0:
        return float("nan")
    high, low = float(frame["HighPx"].max()), float(frame["LowPx"].min())
    if not low > 0:
        return float("nan")
    return max(high - low, abs(high - pre_close), abs(low - pre_close))


def load_days(symbol: str, data_dir: str, atr_days: int) -> list[DayData]:
    """加载一个标的的全部交易日，仅保留连续竞价时段且盘口有效的快照。

    剔除集合竞价（09:25–09:30 开盘、14:57–15:00 收盘）与午休（11:30–13:00）快照：
    前者按单一价格集中撮合，后者为价格冻结、成交量零增量的重复行情，二者均不适用
    逐笔穿价撮合，若混入序列会污染回看窗口并产生无法成交的决策点。
    同时剔除盘口单边为空的快照（涨跌停封板时无对手价，该方向本就无法成交）。

    每日 ATR 严格取此前 atr_days 个完整交易日真实波幅的均值，当日盘中恒定、无前视；
    历史不足或结果无效的交易日 atr 记 nan，由 DayMarket.tradable 排除。
    """
    df = _read_symbol(symbol, data_dir)
    ts = pd.to_datetime(df["MDDate"] + df["MDTime"].str.zfill(9), format="%Y%m%d%H%M%S%f")
    tod = ts.dt.time
    valid = (
        (((tod >= MORNING_OPEN) & (tod <= MORNING_CLOSE))
         | ((tod >= AFTERNOON_OPEN) & (tod < AFTERNOON_CLOSE)))
        & (df["LastPx"] > 0)
        & (df["Buy1Price"] > 0)
        & (df["Sell1Price"] > 0)
    )
    df = df[valid].assign(_ts=ts[valid])
    days: list[DayData] = []
    for date, g in df.groupby("MDDate", sort=True):
        # 稳定排序：同一时间戳的快照保持原始顺序，确保累计量（成交量 / 成交笔数）单调
        g = g.sort_values("_ts", kind="stable").drop(columns="_ts").reset_index(drop=True)
        pre_close = float(g["PreClosePx"].iloc[0])
        days.append(DayData(date=str(date), frame=g, pre_close=pre_close, atr=float("nan")))

    trs = np.asarray([_true_range(d.frame, d.pre_close) for d in days])
    for i, d in enumerate(days):
        if i >= atr_days:
            d.atr = float(trs[i - atr_days : i].mean())
    return days


def walk_forward_splits(
    days: list[DayData], train_ratio: float, val_ratio: float, n_folds: int
) -> list[tuple[list[DayData], list[DayData], list[DayData]]]:
    """滚动前向切分：等长的验证 / 测试窗口逐折后移，训练集为窗口之前的全部交易日。

    末折即按 (train_ratio, val_ratio) 的单次时序切分，其余折依次前移一个测试窗口，
    因此 n_folds = 1 退化为单次切分。各折测试窗口互不重叠、覆盖不同市场状态，
    但相邻折的验证窗口落在前一折的测试窗口内，跨折聚合选参时应视其为同一段行情。
    """
    n = len(days)
    n_val = int(n * val_ratio)
    n_test = n - int(n * train_ratio) - n_val
    splits = []
    for fold in range(n_folds):
        test_end = n - (n_folds - 1 - fold) * n_test
        test_start, val_start = test_end - n_test, test_end - n_test - n_val
        if val_start <= 0:
            raise ValueError(
                f"{n} 个交易日不足以切出 {n_folds} 折（每折验证 {n_val} 日、测试 {n_test} 日）"
            )
        splits.append((days[:val_start], days[val_start:test_start], days[test_start:test_end]))
    return splits
