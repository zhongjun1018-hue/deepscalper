"""特征工程：微观盘口特征、宏观 OHLCV+技术指标、训练标签。

微观特征（每快照 50 维）：
  - 10 档盘口价量（40 维）：各档价格除以一档价格，数量取 log1p(各档量 / 一档量)；
  - 微观结构增强（10 维）：委托笔数、撤单、加权均价、成交量增量等订单流字段。

宏观特征（11 维）：将回看窗口聚合为 30 根 20-tick OHLCV bar，计算 10 个相对价格
指标并补充相对量能指标。

私有状态（7 维，design 5.2）：由 PRIV_RAW_DIM 列的原始记录在 DayMarket.observe
中归一得到，原始列见 PRIV_* 常量。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MICRO_LOB_DIM = 40
MICRO_EXTRA_DIM = 10
MICRO_DIM = MICRO_LOB_DIM + MICRO_EXTRA_DIM
MACRO_DIM = 11
PRIVATE_DIM = 7

# 私有状态的原始记录列（按 tick 保存，归一后喂入 LSTM）
PRIV_POS, PRIV_CASH, PRIV_CENTER, PRIV_WIDTH, PRIV_SIZE, PRIV_LAST_FILL = range(6)
PRIV_RAW_DIM = 6


class FeatureStats:
    """基于训练集拟合的 z-score 标准化统计量（微观 50 维 + 宏观 11 维）。

    仅用训练交易日拟合，验证 / 测试集复用同一统计量，避免未来信息泄漏。
    """

    def __init__(self, micro_mean, micro_std, macro_mean, macro_std, clip: float):
        self.micro_mean, self.micro_std = micro_mean, micro_std
        self.macro_mean, self.macro_std = macro_mean, macro_std
        self.clip = clip

    def micro(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.micro_mean) / self.micro_std, -self.clip, self.clip).astype(np.float32)

    def macro(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.macro_mean) / self.macro_std, -self.clip, self.clip).astype(np.float32)


def fit_feature_stats(markets, cfg) -> FeatureStats:
    """在训练 markets 上拟合特征统计量。

    决策点由事件触发、依策略而变，故宏观统计量在固定 tick 网格 sample_points 上拟合，
    使标准化口径与策略无关；微观特征抽样以控制内存。
    """
    micro_rows, macro_rows = [], []
    for m in markets:
        micro_rows.append(m.micro[::4])
        macro_rows.extend(m.macro_at(t, normalized=False) for t in m.sample_points)
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
            np.log1p(_safe_ratio(ask_q, np.maximum(ask_q[:, :1], 1.0))),
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
    # 累计字段偶有缺失或回退；负增量不是成交流，统一记 0。
    vol = np.maximum(np.diff(tvt, prepend=tvt[0]), 0.0)
    ntr = np.maximum(np.diff(ntr_raw, prepend=ntr_raw[0]), 0.0)
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
    """由回看窗口的中间价 / 成交量序列构建宏观特征 (MACRO_DIM,)。"""
    bars = mid_window.reshape(n_bars, -1)
    closes = bars[:, -1]
    bar_vol = vol_window.reshape(n_bars, -1).sum(axis=1)

    c = closes[-1]                               # 中间价恒为正，无需除零保护
    z_close = c / closes[-2] - 1.0
    feats = [
        bars[-1, 0] / c - 1.0,                   # z_open
        bars[-1].max() / c - 1.0,                # z_high
        bars[-1].min() / c - 1.0,                # z_low
        z_close,                                 # z_close
    ]
    feats += [closes[-k:].mean() / c - 1.0 for k in (5, 10, 15, 20, 25, 30)]  # z_d_k
    feats.append(_safe_ratio(bar_vol[-1], bar_vol.mean()) - 1.0)              # z_volume
    return np.asarray(feats, dtype=np.float32)


def future_price_index(t: int, horizon: int, n: int) -> int:
    """hindsight 标签的未来终点（不跨交易日，尾部截断）。"""
    return min(t + horizon, n - 1)


def volatility_label(mid: np.ndarray, t: int, horizon: int, step: int) -> float:
    """未来 horizon 窗口内逐 step tick 对数收益的标准差（日尾截断）。"""
    idx = np.arange(t, min(t + horizon, len(mid) - 1) + 1, step)
    rets = np.diff(np.log(mid[idx]))
    return float(rets.std()) if rets.size > 1 else 0.0
