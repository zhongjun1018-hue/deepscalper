"""特征工程：微观逐快照序列、宏观 bar 指标与窗口统计、私有状态归一。

微观特征（每快照 66 维，定义见 data/features.md §5.1）：
  - 10 档盘口价量（40 维）：各档价格相对买一价的偏离，数量取 log1p(各档量 / 一档量)；
  - 逐快照微观结构（26 维）：盘口形态、成交流与日内状态——公共特征（§3）中凡为
    「逐快照量的窗口聚合」者，其逐快照原语均以同一定义纳入。

宏观特征（11 + 24 + 5 = 40 维，均进 MLP 通道）：
  - 11 维：将回看窗口聚合为 30 根 20-tick OHLCV bar，计算 10 个相对价格
    指标并补充相对量能指标；
  - 24 维 + 5 维：统一缓存的窗口统计（只取 MACRO_FEATURE_NAMES 一档）与 LightGBM
    前瞻预测（data_provider/windows.py），行索引即 tick 索引，由 DayMarket.macro_at
    取决策 tick 所在行拼接，与 11 维共用同一套标准化统计量。

私有状态（7 维，data/features.md §5.3）：由 PRIV_RAW_DIM 列的原始记录在 DayMarket.observe
中归一得到，原始列见 PRIV_* 常量。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_provider.ticks import top_order_share
from data_provider.windows import FEATURE_NAMES, TARGET_NAMES

MICRO_LOB_DIM = 40
MICRO_EXTRA_DIM = 26
MICRO_DIM = MICRO_LOB_DIM + MICRO_EXTRA_DIM
BAR_DIM = 11

# 已由微观序列逐快照给出的窗口统计（day_pos 由私有序列的剩余时间给出）不进宏观通道，
# 避免同一信息在两条通道重复；宏观只留整段窗口才有定义的量（features.md §5.2）
SEQUENCE_COVERED = frozenset({
    "oc_ret", "vwap_rel", "ofi", "idle_share", "spread", "qi1", "qi_gap", "tqi",
    "depth_rel", "width", "width_asym", "far_press", "quote_rate", "queue_churn",
    "l1_count", "l1_top", "rel_day_open", "dist_up", "dist_dn", "range_pos",
    "day_pos", "session", "gap",
})
MACRO_FEATURE_NAMES = tuple(name for name in FEATURE_NAMES if name not in SEQUENCE_COVERED)
MACRO_FEATURE_COLUMNS = np.array(
    [FEATURE_NAMES.index(name) for name in MACRO_FEATURE_NAMES])

WINDOW_DIM = len(MACRO_FEATURE_NAMES)  # 进宏观通道的窗口统计
PRED_DIM = len(TARGET_NAMES)           # LightGBM 前瞻目标预测（统一缓存的 preds）
MACRO_DIM = BAR_DIM + WINDOW_DIM + PRED_DIM
PRIVATE_DIM = 7

# 私有状态的原始记录列（按 tick 保存，归一后喂入 LSTM）
PRIV_POS, PRIV_CASH, PRIV_CENTER, PRIV_WIDTH, PRIV_SIZE, PRIV_LAST_FILL = range(6)
PRIV_RAW_DIM = 6


class FeatureStats:
    """基于训练集拟合的 z-score 标准化统计量（微观 66 维 + 宏观 40 维）。

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


def _safe_log(price: np.ndarray) -> np.ndarray:
    """价格取对数；缺档或缺字段（非正值）记 0，由调用方的差分口径吸收。"""
    return np.log(np.where(price > 0.0, price, 1.0))


def _quote_changed(bid1: np.ndarray, ask1: np.ndarray) -> np.ndarray:
    """一档报价相对前一快照是否变化（首条快照无前值，记 0）。"""
    changed = (bid1 != np.roll(bid1, 1)) | (ask1 != np.roll(ask1, 1))
    changed[0] = False
    return changed.astype(np.float64)


def _step_churn(price: np.ndarray, qty: np.ndarray) -> np.ndarray:
    """逐快照一档队列变化率 |ΔQ|/Q_prev；价格变动或前值非正时无定义，记 0。"""
    prev_price, prev_qty = np.roll(price, 1), np.roll(qty, 1)
    holding = (price == prev_price) & (prev_qty > 0)
    holding[0] = False
    return np.where(holding, np.abs(qty - prev_qty) / np.where(holding, prev_qty, 1.0), 0.0)


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
    tva = frame["TotalValueTrade"].to_numpy(np.float64)
    ntr_raw = frame["NumTrades"].to_numpy(np.float64)
    # 累计字段偶有缺失或回退；负增量不是成交流，统一记 0。
    vol = np.maximum(np.diff(tvt, prepend=tvt[0]), 0.0)
    amt = np.maximum(np.diff(tva, prepend=tva[0]), 0.0)
    ntr = np.maximum(np.diff(ntr_raw, prepend=ntr_raw[0]), 0.0)
    log_mid = np.log(mid)
    log_ret = np.diff(log_mid, prepend=log_mid[0])

    bid_depth = bid_q.sum(1)
    ask_depth = ask_q.sum(1)
    qi1 = _safe_ratio(bid_q[:, 0] - ask_q[:, 0], bid_q[:, 0] + ask_q[:, 0])
    log_bid1 = _safe_log(bid_p[:, 0])
    log_ask1 = _safe_log(ask_p[:, 0])
    bid_width = log_bid1 - _safe_log(bid_p[:, -1])
    ask_width = _safe_log(ask_p[:, -1]) - log_ask1

    # 成交均价相对前中间价的偏离：vwap_rel / ofi 的逐快照版本
    traded = (vol > 0.0) & (amt > 0.0)
    vwap_dev = np.where(traded, np.log(np.where(traded, amt, 1.0)
                                       / np.where(traded, vol, 1.0))
                        - np.roll(log_mid, 1), 0.0)

    high = frame["HighPx"].to_numpy(np.float64)
    low = frame["LowPx"].to_numpy(np.float64)
    span = high - low
    hour = frame["MDTime"].astype(str).str.zfill(9).str[:2].astype(np.int64).to_numpy()

    extra = np.column_stack(
        [
            log_ask1 - log_bid1,                                                 # 相对价差
            qi1,                                                                 # 一档失衡
            _safe_ratio(bid_depth - ask_depth, bid_depth + ask_depth) - qi1,     # 档位失衡差
            _safe_ratio(tbq - toq, tbq + toq),                                   # 总量失衡
            np.log1p(bid_depth + ask_depth),                                     # 十档深度
            (bid_width + ask_width) / 2.0,                                       # 盘口宽度
            ask_width - bid_width,                                               # 宽度不对称
            _safe_log(frame["WeightedAvgBidPx"].to_numpy(np.float64)) - log_mid,
            _safe_log(frame["WeightedAvgOfferPx"].to_numpy(np.float64)) - log_mid,
            _quote_changed(bid_p[:, 0], ask_p[:, 0]),                            # 报价变化
            0.5 * (_step_churn(bid_p[:, 0], bid_q[:, 0])
                   + _step_churn(ask_p[:, 0], ask_q[:, 0])),                     # 一档队列变化率
            np.log1p(frame["Buy1NumOrders"].to_numpy(np.float64)
                     + frame["Sell1NumOrders"].to_numpy(np.float64)),            # 一档笔数
            top_order_share(frame),                                              # 一档大单占比
            _safe_ratio(nbo - noo, nbo + noo),                                   # 委托笔数失衡
            _safe_ratio(bid_n.sum(1) - ask_n.sum(1), bid_n.sum(1) + ask_n.sum(1)),  # 十档笔数失衡
            _safe_ratio(wb - ws, wb + ws),                                       # 撤单失衡
            np.log1p(vol),                                                       # 成交量增量
            np.log1p(ntr),                                                       # 成交笔数增量
            vwap_dev,                                                            # 成交均价偏离
            np.sign(vwap_dev) * np.log1p(vol),                                   # 有向成交量
            log_ret,                                                             # 中间价对数收益
            log_mid - _safe_log(frame["OpenPx"].to_numpy(np.float64)),           # 日内开盘偏离
            _safe_log(frame["MaxPx"].to_numpy(np.float64)) - log_mid,            # 距涨停
            log_mid - _safe_log(frame["MinPx"].to_numpy(np.float64)),            # 距跌停
            np.where(span > 0.0,                                                 # 日内区间位置
                     np.clip(_safe_ratio(frame["LastPx"].to_numpy(np.float64) - low, span),
                             0.0, 1.0),
                     0.5),
            (hour >= 13).astype(np.float64),                                     # 时段
        ]
    )
    feats = np.concatenate([lob, extra], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_macro_features(mid_window: np.ndarray, vol_window: np.ndarray, n_bars: int) -> np.ndarray:
    """由回看窗口的中间价 / 成交量序列构建宏观特征 (BAR_DIM,)。"""
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
