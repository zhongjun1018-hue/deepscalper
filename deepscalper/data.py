"""数据加载：读取月度 tick parquet，过滤连续竞价时段，按交易日组织。"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import time

import pandas as pd

# A 股连续竞价时段：上午 09:30:00–11:30:00，下午 13:00:00–14:56:59。
# 上午含 11:30:00（该快照为上午收盘瞬时状态），下午不含 14:57:00（收盘集合竞价起点）。
MORNING_OPEN, MORNING_CLOSE = time(9, 30), time(11, 30)
AFTERNOON_OPEN, AFTERNOON_CLOSE = time(13, 0), time(14, 57)

PRICE_COLS = [f"{s}{i}Price" for s in ("Buy", "Sell") for i in range(1, 11)]
QTY_COLS = [f"{s}{i}OrderQty" for s in ("Buy", "Sell") for i in range(1, 11)]
ORDERS_COLS = [f"{s}{i}NumOrders" for s in ("Buy", "Sell") for i in range(1, 11)]

READ_COLS = [
    "MDDate", "MDTime", "LastPx", "TotalVolumeTrade", "NumTrades",
    "TotalBidQty", "TotalOfferQty", "NumBidOrders", "NumOfferOrders",
    "WeightedAvgBidPx", "WeightedAvgOfferPx",
    "WithdrawBuyAmount", "WithdrawSellAmount",
] + PRICE_COLS + QTY_COLS + ORDERS_COLS


@dataclass
class DayData:
    """单个交易日的连续竞价快照序列（已按时间排序、剔除无效行）。"""

    date: str
    frame: pd.DataFrame  # 索引 0..N-1，含 READ_COLS 全部字段


def _read_symbol(symbol: str, data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, symbol, "*_tick.parquet")))
    if not files:
        raise FileNotFoundError(f"未找到 {symbol} 的 tick 数据：{data_dir}/{symbol}")
    return pd.concat((pd.read_parquet(f, columns=READ_COLS) for f in files), ignore_index=True)


def load_days(symbol: str, data_dir: str) -> list[DayData]:
    """加载一个标的的全部交易日，仅保留连续竞价时段且盘口有效的快照。

    剔除集合竞价（09:25–09:30 开盘、14:57–15:00 收盘）与午休（11:30–13:00）快照：
    前者按单一价格集中撮合，后者为价格冻结、成交量零增量的重复行情，二者均不适用
    逐笔穿价撮合，若混入序列会污染回看窗口并产生无法成交的决策点。
    同时剔除盘口单边为空的快照（涨跌停封板时无对手价，该方向本就无法成交）。
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
        days.append(DayData(date=str(date), frame=g))
    return days


def split_days(
    days: list[DayData], train_ratio: float, val_ratio: float
) -> tuple[list[DayData], list[DayData], list[DayData]]:
    """按交易日时序 7:1:2 切分训练 / 验证 / 测试集。"""
    n = len(days)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return days[:n_train], days[n_train : n_train + n_val], days[n_train + n_val :]
