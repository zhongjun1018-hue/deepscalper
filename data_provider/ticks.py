"""tick 数据加载：全项目唯一入口（月度 parquet → 连续竞价日序列，含逐日 ATR）。"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from strategy.width import daily_bars, true_range

# A 股连续竞价时段：上午 09:30:00–11:30:00，下午 13:00:00–14:56:59。
# 上午含 11:30:00（该快照为上午收盘瞬时状态），下午不含 14:57:00（收盘集合竞价起点）。
MORNING_OPEN, MORNING_CLOSE = time(9, 30), time(11, 30)
AFTERNOON_OPEN, AFTERNOON_CLOSE = time(13, 0), time(14, 57)

# 连续竞价压缩分钟网格：09:30–11:30 为分钟 0–119，13:00–14:57 为分钟 120–236（午休压缩）
MORNING_MINUTES = 120
AFTERNOON_MINUTES = 117
MINUTES_PER_DAY = MORNING_MINUTES + AFTERNOON_MINUTES

_MORNING_OPEN_MIN = 9 * 60 + 30   # 分钟计的时段起点（minute_index 用）
_AFTERNOON_OPEN_MIN = 13 * 60

PRICE_COLS = [f"{s}{i}Price" for s in ("Buy", "Sell") for i in range(1, 11)]
QTY_COLS = [f"{s}{i}OrderQty" for s in ("Buy", "Sell") for i in range(1, 11)]
ORDERS_COLS = [f"{s}{i}NumOrders" for s in ("Buy", "Sell") for i in range(1, 11)]

# 合并窗口特征构建所需的全部字段（含十档行情、一档订单明细与逐日统计）
READ_COLS = [
    "MDDate", "MDTime",
    "LastPx", "PreClosePx", "OpenPx", "HighPx", "LowPx", "MaxPx", "MinPx",
    "TotalVolumeTrade", "TotalValueTrade", "NumTrades",
    "TotalBidQty", "TotalOfferQty", "NumBidOrders", "NumOfferOrders",
    "WeightedAvgBidPx", "WeightedAvgOfferPx",
    "WithdrawBuyAmount", "WithdrawSellAmount",
    "Buy1OrderDetail", "Sell1OrderDetail",
] + PRICE_COLS + QTY_COLS + ORDERS_COLS


@dataclass
class DayData:
    """单个交易日的连续竞价快照序列（已按时间排序、去重并剔除无效行）。"""

    date: str
    frame: pd.DataFrame  # 索引 0..N-1，含 READ_COLS 全部字段
    pre_close: float     # 前收盘价
    open_px: float       # 当日开盘价（集合竞价）
    atr: float           # 前 atr_days 个完整交易日真实波幅的均值（价格计）；历史不足记 nan


def symbol_files(data_dir: str, symbol: str) -> list[str]:
    """标的的全部 tick 源文件（按文件名排序）。"""
    return sorted(glob.glob(os.path.join(data_dir, symbol, "*_tick.parquet")))


def list_symbols(data_dir: str = "data") -> list[str]:
    """data_dir 下含 tick 源文件的全部标的（排序）；各入口 --symbols 缺省时的标的全集。"""
    return sorted(name for name in os.listdir(data_dir)
                  if symbol_files(data_dir, name))


def symbol_source_signature(data_dir: str, symbol: str) -> list[list]:
    """源文件的廉价身份标识（文件名 / 大小 / mtime），供缓存失效判定。"""
    return [
        [os.path.basename(path), os.path.getsize(path), os.stat(path).st_mtime_ns]
        for path in symbol_files(data_dir, symbol)
    ]


def minute_index(mdtime: pd.Series) -> np.ndarray:
    """HHMMSSmmm → 压缩分钟索引（0..236）；11:30:00 收盘快照归入分钟 119，网格外记 -1。"""
    text = mdtime.astype(str).str.zfill(9)
    total = text.str[:2].astype(np.int64) * 60 + text.str[2:4].astype(np.int64)
    morning = (total - _MORNING_OPEN_MIN).to_numpy()
    afternoon = (MORNING_MINUTES + total - _AFTERNOON_OPEN_MIN).to_numpy()
    idx = np.where(
        (morning >= 0) & (morning <= MORNING_MINUTES),
        np.minimum(morning, MORNING_MINUTES - 1), -1,
    )
    return np.where((afternoon >= MORNING_MINUTES) & (afternoon < MINUTES_PER_DAY), afternoon, idx)


def _top_share(detail: pd.Series, total_quantity: np.ndarray) -> np.ndarray:
    """一侧一档订单明细中最大单笔委托占一档总量的比例。"""
    parts = pd.to_numeric(detail.str[1:-1].str.split("|").explode(), errors="coerce")
    largest = parts.groupby(level=0).max().reindex(detail.index).to_numpy(dtype=np.float64)
    return np.divide(largest, total_quantity, out=np.full(len(largest), np.nan),
                     where=total_quantity > 0)


def top_order_share(frame: pd.DataFrame) -> np.ndarray:
    """逐快照一档大单占比：已披露订单中最大单笔委托占一档总量的比例（买卖均值）。"""
    return 0.5 * (
        _top_share(frame["Buy1OrderDetail"], frame["Buy1OrderQty"].to_numpy(np.float64)) +
        _top_share(frame["Sell1OrderDetail"], frame["Sell1OrderQty"].to_numpy(np.float64)))


def minute_labels() -> np.ndarray:
    """237 个压缩分钟的开盘时刻标签（'0930'…'1456'，webviz 横轴用）。"""
    starts = [9 * 60 + 30 + minute for minute in range(MORNING_MINUTES)]
    starts += [13 * 60 + minute for minute in range(MINUTES_PER_DAY - MORNING_MINUTES)]
    return np.array(["{:02d}{:02d}".format(value // 60, value % 60) for value in starts])


def load_days(symbol: str, data_dir: str = "data", atr_days: int = 3) -> list[DayData]:
    """加载一个标的的全部交易日，仅保留连续竞价时段且盘口有效的快照。

    剔除集合竞价（09:25–09:30 开盘、14:57–15:00 收盘）与午休（11:30–13:00）快照：
    前者按单一价格集中撮合，后者为价格冻结、成交量零增量的重复行情，二者均不适用
    逐笔穿价撮合，若混入序列会污染回看窗口并产生无法成交的决策点。
    同时剔除盘口单边为空的快照（涨跌停封板时无对手价，该方向本就无法成交），
    并按 (MDDate, MDTime) 去重（keep="last"，源文件后写覆盖先写的重复快照）。

    每日 ATR 严格取此前 atr_days 个完整交易日真实波幅的均值（TR 口径见
    strategy/width.py），当日盘中恒定、无前视；历史不足或 TR 无效的交易日 atr 记 nan。
    """
    files = symbol_files(data_dir, symbol)
    if not files:
        raise FileNotFoundError(f"未找到 {symbol} 的 tick 数据：{data_dir}/{symbol}")
    df = pd.concat((pd.read_parquet(f, columns=READ_COLS) for f in files), ignore_index=True)

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
    df = df.drop_duplicates(subset=["MDDate", "MDTime"], keep="last")

    days: list[DayData] = []
    for date, g in df.groupby("MDDate", sort=True):
        # 稳定排序：同一时间戳的快照保持原始顺序，确保累计量（成交量 / 成交笔数）单调
        g = g.sort_values("_ts", kind="stable").drop(columns="_ts").reset_index(drop=True)
        days.append(DayData(date=str(date), frame=g,
                            pre_close=float(g["PreClosePx"].iloc[0]),
                            open_px=float(g["OpenPx"].iloc[0]), atr=float("nan")))

    # ATR 与网格宽度共用同一 TR 实现；逐日 K 线无效的日期 TR 记 nan，使随后窗口亦失效
    tr = true_range(daily_bars(df)).reindex([d.date for d in days])
    atr = tr.rolling(atr_days).mean().shift(1).to_numpy()
    for d, a in zip(days, atr):
        d.atr = float(a)
    return days
