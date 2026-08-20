"""特征工程：微观逐快照序列、宏观 bar 指标与窗口统计、私有状态归一。

微观特征（每快照 66 维，定义见 data/features.md §5.1）：
  - 10 档盘口价量（40 维）：各档价格相对买一价的偏离，数量取 log1p(各档量 / 一档量)；
  - 逐快照微观结构（26 维）：盘口形态、成交流与日内状态——公共特征（§3）中凡为
    「逐快照量的窗口聚合」者，其逐快照原语均以同一定义纳入。

宏观特征（11 + 24 + 5 = 40 维，均进 MLP 通道）：
  - 11 维：回看窗口的 30 根 1 分钟 OHLCV bar（DayMarket 按分钟网格预聚合），
    计算 10 个相对价格指标并补充相对量能指标；
  - 24 维 + 5 维：统一缓存的窗口统计（只取 MACRO_FEATURE_NAMES 一档）与 LightGBM
    前瞻预测（data_provider/windows.py），行索引即分钟索引，由 DayMarket.macro_at
    取决策锚点所在行拼接，与 11 维共用同一套标准化统计量。

私有状态（11 维，data/features.md §5.3）：由 PRIV_RAW_DIM 列的原始记录在
DayMarket.observe 中归一得到（含决策区间内成交过程的 4 个通道），原始列见
PRIV_* 常量。
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
PRIVATE_DIM = 11

# 私有状态的原始记录列（按 tick 保存，锚点抽样后归一喂入 LSTM）：
# 前 6 列为账户与网格状态；后 6 列服务决策区间内成交过程的 4 个通道——
# 累计买卖笔数（相邻锚点差分即步内笔数）与当前区间的成交统计
#（起点中间价 p_dec、最近成交价、相对 p_dec 的带符号累计成交额与累计手数）。
(PRIV_POS, PRIV_CASH, PRIV_CENTER, PRIV_WIDTH, PRIV_SIZE, PRIV_LAST_FILL,
 PRIV_CUM_BUYS, PRIV_CUM_SELLS, PRIV_DEC_MID, PRIV_LAST_PX,
 PRIV_INT_NOTIONAL, PRIV_INT_LOTS) = range(12)
PRIV_RAW_DIM = 12


class FeatureStats:
    """逐标的在训练集上拟合的 z-score 标准化统计量（微观 66 维 + 宏观 40 维）。

    各数组形状为 (n_symbols, dim)，行索引即 symbol_id（排序后标的集合中的索引）：
    每个标的只用自己的训练交易日拟合，验证 / 测试集复用同一统计量，避免未来信息
    泄漏；统计量随检查点保存，回放侧不重新拟合。
    """

    def __init__(self, micro_mean, micro_std, macro_mean, macro_std, clip: float):
        self.micro_mean, self.micro_std = micro_mean, micro_std
        self.macro_mean, self.macro_std = macro_mean, macro_std
        self.clip = clip

    def micro(self, x: np.ndarray, symbol_id: int) -> np.ndarray:
        return np.clip((x - self.micro_mean[symbol_id]) / self.micro_std[symbol_id],
                       -self.clip, self.clip).astype(np.float32)

    def macro(self, x: np.ndarray, symbol_id: int) -> np.ndarray:
        return np.clip((x - self.macro_mean[symbol_id]) / self.macro_std[symbol_id],
                       -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict:
        """检查点序列化：均值 / 标准差存为张量（torch.load 默认只接受权重类对象）。"""
        import torch

        return {name: torch.as_tensor(getattr(self, name), dtype=torch.float64)
                for name in ("micro_mean", "micro_std", "macro_mean", "macro_std")
                } | {"clip": self.clip}

    @classmethod
    def from_state_dict(cls, state: dict) -> "FeatureStats":
        return cls(*(np.asarray(state[name]) for name in
                     ("micro_mean", "micro_std", "macro_mean", "macro_std")),
                   clip=state["clip"])


class _Moments:
    """流式均值 / 标准差累加器：拟合无需在内存中拼接全部样本行。"""

    def __init__(self):
        self.count = 0
        self.total = self.squares = 0.0

    def add(self, rows: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=np.float64)
        self.count += len(rows)
        self.total = self.total + rows.sum(axis=0)
        self.squares = self.squares + (rows ** 2).sum(axis=0)

    def stats(self) -> tuple[np.ndarray, np.ndarray]:
        mean = self.total / self.count
        variance = np.maximum(self.squares / self.count - mean ** 2, 0.0)
        return mean, np.maximum(np.sqrt(variance), 1e-8)


def fit_feature_stats(markets, cfg) -> FeatureStats:
    """在训练 markets 上逐标的拟合特征统计量（行索引即 symbol_id）。

    宏观统计量在固定分钟锚点网格 sample_points 上拟合，与决策相位无关（训练起点
    随机偏移不影响标准化口径）；微观特征抽样以控制内存。
    """
    micro = [_Moments() for _ in range(cfg.n_symbols)]
    macro = [_Moments() for _ in range(cfg.n_symbols)]
    for m in markets:
        micro[m.symbol_id].add(m.micro[::4])
        macro[m.symbol_id].add([m.macro_at(t, normalized=False) for t in m.sample_points])
    micro_stats = [moments.stats() for moments in micro]
    macro_stats = [moments.stats() for moments in macro]
    return FeatureStats(np.stack([mean for mean, _ in micro_stats]),
                        np.stack([std for _, std in micro_stats]),
                        np.stack([mean for mean, _ in macro_stats]),
                        np.stack([std for _, std in macro_stats]),
                        clip=cfg.norm_clip)


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


def build_macro_features(bars: np.ndarray) -> np.ndarray:
    """由回看窗口的分钟 bar 序列构建宏观特征 (BAR_DIM,)。

    bars 为 (n_bars, 5) 的 [open, high, low, close, volume]（中间价 OHLC，
    无快照分钟已按前收盘价填充、量能记 0，见 DayMarket）。
    """
    closes = bars[:, 3]
    volumes = bars[:, 4]
    c = closes[-1]                               # 中间价恒为正，无需除零保护
    feats = [
        bars[-1, 0] / c - 1.0,                   # z_open
        bars[-1, 1] / c - 1.0,                   # z_high
        bars[-1, 2] / c - 1.0,                   # z_low
        c / closes[-2] - 1.0,                    # z_close
    ]
    feats += [closes[-k:].mean() / c - 1.0 for k in (5, 10, 15, 20, 25, 30)]  # z_d_k
    feats.append(_safe_ratio(volumes[-1], volumes.mean()) - 1.0)              # z_volume
    return np.asarray(feats, dtype=np.float32)
