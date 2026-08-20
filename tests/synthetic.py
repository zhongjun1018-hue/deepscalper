"""测试共用的合成交易日：字段覆盖 data_provider.ticks.READ_COLS 中特征构建所需的列。"""

import numpy as np
import pandas as pd

from data_provider.ticks import DayData


def synthetic_day(date: str = "20260102", n: int = 650) -> DayData:
    """合成连续竞价日：09:30 起每 300ms 一条快照，盘口、累计量与日内状态字段齐备。"""
    frame = pd.DataFrame({
        "MDTime": pd.date_range("2026-01-02 09:30:00", periods=n,
                                freq="300ms").strftime("%H%M%S%f").str[:-3],
        "LastPx": 10.0, "OpenPx": 10.0, "HighPx": 10.2, "LowPx": 9.8,
        "MaxPx": 11.0, "MinPx": 9.0,
        "TotalVolumeTrade": np.arange(n, dtype=np.float64) * 100.0,
        "TotalValueTrade": np.arange(n, dtype=np.float64) * 1000.0,
        "NumTrades": np.arange(n, dtype=np.float64) + 1.0,
        "TotalBidQty": 1e5, "TotalOfferQty": 1e5,
        "NumBidOrders": 100.0, "NumOfferOrders": 100.0,
        "WithdrawBuyAmount": 0.0, "WithdrawSellAmount": 0.0,
        "WeightedAvgBidPx": 9.99, "WeightedAvgOfferPx": 10.01,
        "Buy1NumOrders": 5.0, "Sell1NumOrders": 5.0,
        "Buy1OrderDetail": "[600|400]", "Sell1OrderDetail": "[600|400]",
    })
    for level in range(1, 11):
        frame[f"Buy{level}Price"] = 10.0 - level * 0.01
        frame[f"Sell{level}Price"] = 10.0 + level * 0.01
        frame[f"Buy{level}OrderQty"] = 1000.0
        frame[f"Sell{level}OrderQty"] = 1000.0
        frame[f"Buy{level}NumOrders"] = 10.0
        frame[f"Sell{level}NumOrders"] = 10.0
    return DayData(date=date, frame=frame, pre_close=10.0, open_px=10.0, atr=0.05)
