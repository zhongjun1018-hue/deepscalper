"""数据加载：读取月度 tick parquet，过滤连续竞价时段，按交易日组织。"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

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

    def __len__(self) -> int:
        return len(self.frame)


def _read_symbol(symbol: str, data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, symbol, "*_tick.parquet")))
    if not files:
        raise FileNotFoundError(f"未找到 {symbol} 的 tick 数据：{data_dir}/{symbol}")
    return pd.concat((pd.read_parquet(f, columns=READ_COLS) for f in files), ignore_index=True)


def load_days(symbol: str, data_dir: str = "data") -> list[DayData]:
    """加载一个标的的全部交易日，仅保留 09:30-15:00 且盘口有效的快照。"""
    df = _read_symbol(symbol, data_dir)
    ts = pd.to_datetime(df["MDDate"] + df["MDTime"].str.zfill(9), format="%Y%m%d%H%M%S%f")
    tod = ts.dt.time
    valid = (
        (tod >= pd.Timestamp("09:30").time())
        & (tod <= pd.Timestamp("15:00").time())
        & (df["LastPx"] > 0)
        & (df["Buy1Price"] > 0)
        & (df["Sell1Price"] > 0)
    )
    df = df[valid].assign(_ts=ts[valid])
    days: list[DayData] = []
    for date, g in df.groupby("MDDate", sort=True):
        g = g.sort_values("_ts").drop(columns="_ts").reset_index(drop=True)
        days.append(DayData(date=str(date), frame=g))
    return days


def split_days(
    days: list[DayData], train_ratio: float = 0.7, val_ratio: float = 0.1
) -> tuple[list[DayData], list[DayData], list[DayData]]:
    """按交易日时序 7:1:2 切分训练 / 验证 / 测试集。"""
    n = len(days)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return days[:n_train], days[n_train : n_train + n_val], days[n_train + n_val :]
