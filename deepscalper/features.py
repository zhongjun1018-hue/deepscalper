"""特征工程：微观盘口特征、宏观 OHLCV+技术指标、训练标签。

微观特征（每快照 50 维）：
  - 10 档盘口价量（40 维）：各档价格除以一档价格（论文 5.4），
    数量取 log1p(各档量 / 一档量) 以稳定量纲；
  - 微观结构增强（10 维）：利用 tick 数据的委托笔数、撤单、加权均价、
    成交量增量等字段，刻画买卖力量对比与订单流。

宏观特征（每决策步 12 维）：将回看窗口聚合为 30 根 20-tick OHLCV bar，
按论文 Table 2 计算 11 个相对价格指标，并补充相对量能指标。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MICRO_LOB_DIM = 40
MICRO_EXTRA_DIM = 10
MICRO_DIM = MICRO_LOB_DIM + MICRO_EXTRA_DIM
MACRO_DIM = 12
PRIVATE_DIM = 3  # 归一化持仓、归一化现金、剩余时间比例


class FeatureStats:
    """基于训练集拟合的 z-score 标准化统计量（微观 50 维 + 宏观 12 维）。

    仅用训练交易日拟合，验证 / 测试集复用同一统计量，避免未来信息泄漏。
    """

    def __init__(self, micro_mean, micro_std, macro_mean, macro_std, clip: float = 10.0):
        self.micro_mean, self.micro_std = micro_mean, micro_std
        self.macro_mean, self.macro_std = macro_mean, macro_std
        self.clip = clip

    def micro(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.micro_mean) / self.micro_std, -self.clip, self.clip).astype(np.float32)

    def macro(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.macro_mean) / self.macro_std, -self.clip, self.clip).astype(np.float32)


def fit_feature_stats(markets, cfg) -> FeatureStats:
    """在训练 markets 上拟合特征统计量（微观抽样以控制内存）。"""
    micro_rows, macro_rows = [], []
    for m in markets:
        micro_rows.append(m.micro[::4])
        macro_rows.extend(m.macro_at(t, normalized=False) for t in m.decision_points)
    micro_all = np.concatenate(micro_rows).astype(np.float64)
    macro_all = np.asarray(macro_rows, dtype=np.float64)
    return FeatureStats(
        micro_all.mean(0), np.maximum(micro_all.std(0), 1e-8),
        macro_all.mean(0), np.maximum(macro_all.std(0), 1e-8),
        clip=cfg.norm_clip,
    )


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return num / np.where(np.abs(den) < 1e-12, 1.0, den)


def build_micro_matrix(frame: pd.DataFrame) -> np.ndarray:
    """构建单日微观特征矩阵 (N, MICRO_DIM)，float32。"""
    bid_p = frame[[f"Buy{i}Price" for i in range(1, 11)]].to_numpy(np.float64)
    ask_p = frame[[f"Sell{i}Price" for i in range(1, 11)]].to_numpy(np.float64)
    bid_q = frame[[f"Buy{i}OrderQty" for i in range(1, 11)]].to_numpy(np.float64)
    ask_q = frame[[f"Sell{i}OrderQty" for i in range(1, 11)]].to_numpy(np.float64)
    bid_n = frame[[f"Buy{i}NumOrders" for i in range(1, 11)]].to_numpy(np.float64)
    ask_n = frame[[f"Sell{i}NumOrders" for i in range(1, 11)]].to_numpy(np.float64)

    bid1p = bid_p[:, :1]
    mid = (bid_p[:, 0] + ask_p[:, 0]) / 2.0

    # 10 档价量：价格相对一档价，数量 log1p 相对一档量
    lob = np.concatenate(
        [
            _safe_ratio(bid_p, bid1p) - 1.0,
            _safe_ratio(ask_p, bid1p) - 1.0,
            np.log1p(_safe_ratio(bid_q, np.maximum(bid_q[:, :1], 1.0))),
            np.log1p(_safe_ratio(ask_q, np.maximum(bid_q[:, :1], 1.0))),
        ],
        axis=1,
    )

    tbq = frame["TotalBidQty"].to_numpy(np.float64)
    toq = frame["TotalOfferQty"].to_numpy(np.float64)
    nbo = frame["NumBidOrders"].to_numpy(np.float64)
    noo = frame["NumOfferOrders"].to_numpy(np.float64)
    wb = frame["WithdrawBuyAmount"].to_numpy(np.float64)
    ws = frame["WithdrawSellAmount"].to_numpy(np.float64)
    tvt = frame["TotalVolumeTrade"].to_numpy(np.float64)
    ntr_raw = frame["NumTrades"].to_numpy(np.float64)
    vol = np.diff(tvt, prepend=tvt[0])      # 首拍增量记 0（累计量含开盘竞价，无法回溯）
    ntr = np.diff(ntr_raw, prepend=ntr_raw[0])
    log_ret = np.diff(np.log(mid), prepend=np.log(mid[0]))

    extra = np.column_stack(
        [
            (ask_p[:, 0] - bid_p[:, 0]) / mid,                                  # 相对价差
            _safe_ratio(tbq - toq, tbq + toq),                                   # 总量失衡
            _safe_ratio(nbo - noo, nbo + noo),                                   # 委托笔数失衡
            _safe_ratio(frame["WeightedAvgBidPx"].to_numpy(np.float64), mid) - 1.0,
            _safe_ratio(frame["WeightedAvgOfferPx"].to_numpy(np.float64), mid) - 1.0,
            _safe_ratio(wb - ws, wb + ws),                                       # 撤单失衡
            _safe_ratio(bid_n.sum(1) - ask_n.sum(1), bid_n.sum(1) + ask_n.sum(1)),  # 档位笔数失衡
            np.log1p(vol),
            np.log1p(ntr),
            log_ret * 1e4,                                                       # 中间价对数收益（bp）
        ]
    )
    feats = np.concatenate([lob, extra], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_macro_features(mid_window: np.ndarray, vol_window: np.ndarray, n_bars: int) -> np.ndarray:
    """由回看窗口的中间价 / 成交量序列构建宏观特征 (MACRO_DIM,)。

    mid_window / vol_window 长度为 lookback_ticks，按 bar 聚合后计算
    论文 Table 2 的 11 个指标，并附加相对量能 z_volume。
    """
    closes = mid_window.reshape(n_bars, -1)[:, -1]
    opens = mid_window.reshape(n_bars, -1)[:, 0]
    highs = mid_window.reshape(n_bars, -1).max(axis=1)
    lows = mid_window.reshape(n_bars, -1).min(axis=1)
    bar_vol = vol_window.reshape(n_bars, -1).sum(axis=1)

    c = closes[-1]
    eps = 1e-12
    feats = [
        opens[-1] / c - 1.0,                     # z_open
        highs[-1] / c - 1.0,                     # z_high
        lows[-1] / c - 1.0,                      # z_low
        c / (closes[-2] + eps) - 1.0,            # z_close
        c / (closes[-2] + eps) - 1.0,            # z_adj_close（无复权数据，同 z_close）
    ]
    for k in (5, 10, 15, 20, 25, 30):            # z_d_k
        feats.append(closes[-k:].mean() / (c + eps) - 1.0)
    feats.append(bar_vol[-1] / (bar_vol.mean() + eps) - 1.0)  # z_volume
    return np.asarray(feats, dtype=np.float32)


def future_price_index(t: int, horizon: int, n: int) -> int:
    """hindsight 标签的未来终点（不跨交易日，尾部截断）。"""
    return min(t + horizon, n - 1)


def volatility_label(mid: np.ndarray, t: int, horizon: int, step: int) -> float:
    """未来 horizon 窗口内逐决策步对数收益的标准差（日尾截断）。

    论文 4.4 的 y_vol = σ(r)，r 为「每个时间步」的收益——时间步即决策步
    （论文 1 分钟）；σ(r) 与逐决策步收益同量级，是 η=1 有效的依据（论文 6.4）。
    """
    idx = np.arange(t, min(t + horizon, len(mid) - 1) + 1, step)
    rets = np.diff(np.log(mid[idx]))
    return float(rets.std()) if rets.size > 1 else 0.0
