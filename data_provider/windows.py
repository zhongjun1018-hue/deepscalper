"""标的统一缓存：47 维回看特征、5 维前瞻目标与 5 维预测的唯一构建 / 读写入口。

行索引即 tick 索引：行 t 的输入窗口为当日第 t−L+1 至 t 条快照（L=lookback），
前瞻目标覆盖 (t, t+H]（H=pred），因此任一 tick 都可作为决策点、直接取第 t 行。
回看窗口不完整的行（t<L−1）与前瞻越界的行（t+H 超出当日末快照）记 NaN；缓存按标的
单日最大快照数对齐，尾部行同样记 NaN。

依赖方向说明：特征/标签构建依赖网格引擎回放（data_provider → strategy）是既有设计。

cache/<symbol>.npz 为 forecast 与 control 共用的单一缓存，不按任务拆分：窗口块
{dates, features, targets, width} 由本模块构建，preds 先置 NaN，forecast.train
训练后经 write_predictions 回写。metadata 的窗口级字段（schema、源文件签名、窗口
参数、FEATURE_NAMES / TARGET_NAMES）任一变化即整体重建，预测块随之失效并由
forecast.train.ensure_predictions 重训回写。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from data_provider.ticks import (load_days, symbol_source_signature,
                                 top_order_share)
from strategy import engine as grid_engine
from strategy.width import daily_bars, grid_width

CACHE_SCHEMA = 8


@dataclass(frozen=True)
class WindowSpec:
    """缓存口径的唯一来源：forecast 与 control 的 Config 均内嵌本规格，避免双份参数漂移。"""
    lookback_ticks: int = 600    # 回看窗口（tick）
    pred_ticks: int = 300        # 前瞻窗口（tick）
    bar_ticks: int = 20          # 聚合 bar 长度：当前 bar 量能与 bar 边界跳空（tick）
    atr_mult: float = 0.1        # 网格固定半宽 = atr_mult × ATR
    atr_window: int = 3          # ATR 回溯的完整交易日数
    min_width_ratio: float = 1e-3  # 半宽下限（相对价格）

LEVELS = 10
RESID_QUANTILE = 0.9

PRICE_NAMES = ["oc_ret", "upper", "lower", "vwap_rel"]
PATH_NAMES = ["rv", "path_len", "range_rel", "resid_abs_mean", "resid_abs_q90",
              "abs_slope", "ret2", "er", "rev_rate", "ac1", "semivar_asym", "jump"]
TRADE_NAMES = ["amount", "trade_size", "ofi", "kyle", "idle_share", "trade_conc"]
BOOK_NAMES = ["spread", "spread_cv", "qi1", "qi_gap", "tqi", "depth_rel", "depth_cv",
              "width", "width_asym", "far_press", "quote_rate", "queue_churn",
              "l1_count", "l1_top"]
DAY_NAMES = ["rel_day_open", "vol_rel", "dist_up", "dist_dn", "range_pos", "day_pos",
             "session", "gap"]
GRID_NAMES = ["buy_count", "sell_count", "abs_exposure"]
FEATURE_NAMES = PRICE_NAMES + PATH_NAMES + TRADE_NAMES + BOOK_NAMES + DAY_NAMES + GRID_NAMES
TARGET_NAMES = ["path_len", "rv", "range_rel", "resid_abs_q90", "abs_slope"]
CACHE_ARRAYS = ["dates", "features", "targets", "preds", "width"]


def _divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(numerator, denominator,
                     out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan),
                     where=np.isfinite(denominator) & (denominator != 0))


def path_stats(paths):
    """对数中间价路径（行）的路径统计量；输入须为 [n, ticks] 且至少两个 tick。"""
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths[None, :]
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must have shape [n, ticks] with at least two ticks")

    returns = np.diff(paths, axis=1)
    count = returns.shape[1]
    rv2 = np.sum(returns ** 2, axis=1)
    rv = np.sqrt(rv2)
    path_len = np.sum(np.abs(returns), axis=1)
    range_rel = np.max(paths, axis=1) - np.min(paths, axis=1)

    position = np.arange(paths.shape[1], dtype=np.float64)
    centered_position = position - position.mean()
    centered_path = paths - paths.mean(axis=1, keepdims=True)
    slope = centered_path @ centered_position / np.sum(centered_position ** 2)
    residual = np.abs(centered_path - slope[:, None] * centered_position)
    resid_abs_mean = residual.mean(axis=1)
    delta = paths[:, -1] - paths[:, 0]

    if count >= 2:
        adjacent = returns[:, 1:] * returns[:, :-1]
        nonzero_adjacent = adjacent != 0
        rev_rate = _divide(np.sum(adjacent < 0, axis=1), np.sum(nonzero_adjacent, axis=1))
        mean_return = returns.mean(axis=1, keepdims=True)
        centered_return = returns - mean_return
        ac1 = _divide(np.sum(centered_return[:, 1:] * centered_return[:, :-1], axis=1),
                      np.sum(centered_return ** 2, axis=1))
        bipower = (np.pi / 2.0) * count / (count - 1.0) * np.sum(np.abs(adjacent), axis=1)
    else:
        rev_rate = np.full(len(paths), np.nan)
        ac1 = np.full(len(paths), np.nan)
        bipower = np.full(len(paths), np.nan)

    stats = {
        "rv": rv,
        "path_len": path_len,
        "range_rel": range_rel,
        "resid_abs_mean": resid_abs_mean,
        "resid_abs_q90": np.quantile(residual, RESID_QUANTILE, axis=1),
        "abs_slope": np.abs(slope) * count,
        "ret2": delta ** 2,
        "er": _divide(np.abs(delta), path_len),
        "rev_rate": rev_rate,
        "ac1": ac1,
        "semivar_asym": _divide(np.sum(returns ** 2 * np.sign(returns), axis=1), rv2),
        "jump": _divide(np.maximum(rv2 - bipower, 0.0), rv2),
    }
    invalid = ~np.isfinite(paths).all(axis=1)
    for values in stats.values():
        values[invalid] = np.nan
    return stats


def _positive(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > 0, values, np.nan)


def _nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    return values[finite].mean() if finite.any() else np.nan


def _nanstd(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    return values[finite].std(ddof=0) if finite.any() else np.nan


class _WindowContext:
    """单日全部快照的列式视图：窗口特征所需的全部数组只取一次。"""

    def __init__(self, frame):
        bid1 = frame["Buy1Price"].to_numpy(np.float64)
        ask1 = frame["Sell1Price"].to_numpy(np.float64)
        self.log_mid = np.log(np.where((bid1 > 0) & (ask1 > 0), 0.5 * (bid1 + ask1), np.nan))
        self.lastpx = _positive(frame["LastPx"].to_numpy(np.float64))
        self.trades = frame["NumTrades"].to_numpy(np.float64)
        self.volume = frame["TotalVolumeTrade"].to_numpy(np.float64)
        self.amount = frame["TotalValueTrade"].to_numpy(np.float64)
        self.afternoon = (frame["MDTime"].astype(str).str.zfill(9).str[:2]
                          .astype(np.int64).to_numpy() >= 13)

        self.bid_qty = frame[["Buy{}OrderQty".format(level)
                              for level in range(1, LEVELS + 1)]].to_numpy(np.float64)
        self.ask_qty = frame[["Sell{}OrderQty".format(level)
                              for level in range(1, LEVELS + 1)]].to_numpy(np.float64)
        bid_px = frame[["Buy{}Price".format(level)
                        for level in range(1, LEVELS + 1)]].to_numpy(np.float64)
        ask_px = frame[["Sell{}Price".format(level)
                        for level in range(1, LEVELS + 1)]].to_numpy(np.float64)
        self.bid1 = bid_px[:, 0]
        self.ask1 = ask_px[:, 0]
        log_bid = np.log(_positive(bid_px))
        log_ask = np.log(_positive(ask_px))

        bid_depth = self.bid_qty.sum(axis=1)
        ask_depth = self.ask_qty.sum(axis=1)
        depth = bid_depth + ask_depth
        self.book = {
            "spread": log_ask[:, 0] - log_bid[:, 0],
            "qi1": _divide(self.bid_qty[:, 0] - self.ask_qty[:, 0],
                           self.bid_qty[:, 0] + self.ask_qty[:, 0]),
            "qi_gap": (_divide(bid_depth - ask_depth, depth) -
                       _divide(self.bid_qty[:, 0] - self.ask_qty[:, 0],
                               self.bid_qty[:, 0] + self.ask_qty[:, 0])),
            "tqi": _divide(frame["TotalBidQty"].to_numpy(np.float64) -
                           frame["TotalOfferQty"].to_numpy(np.float64),
                           frame["TotalBidQty"].to_numpy(np.float64) +
                           frame["TotalOfferQty"].to_numpy(np.float64)),
            "depth": depth,
            "width": ((log_bid[:, 0] - log_bid[:, -1]) +
                      (log_ask[:, -1] - log_ask[:, 0])) / 2.0,
            "width_asym": ((log_ask[:, -1] - log_ask[:, 0]) -
                           (log_bid[:, 0] - log_bid[:, -1])),
            "far_press": ((np.log(_positive(frame["WeightedAvgBidPx"].to_numpy(np.float64))) +
                           np.log(_positive(frame["WeightedAvgOfferPx"].to_numpy(np.float64)))) / 2.0 -
                          self.log_mid),
            "l1_count": (frame["Buy1NumOrders"].to_numpy(np.float64) +
                         frame["Sell1NumOrders"].to_numpy(np.float64)),
            "l1_top": frame["l1_top"].to_numpy(np.float64),
        }
        self.open_px = frame["OpenPx"].to_numpy(np.float64)
        self.high_px = frame["HighPx"].to_numpy(np.float64)
        self.low_px = frame["LowPx"].to_numpy(np.float64)
        self.max_px = frame["MaxPx"].to_numpy(np.float64)
        self.min_px = frame["MinPx"].to_numpy(np.float64)


def _price_features(context, start, end):
    path = context.log_mid[start:end]
    volume = context.volume[end - 1] - context.volume[start]
    amount = context.amount[end - 1] - context.amount[start]
    vwap = np.log(amount / volume) - path[-1] if amount > 0 and volume > 0 else np.nan
    return [path[-1] - path[0],
            np.max(path) - max(path[0], path[-1]),
            min(path[0], path[-1]) - np.min(path),
            vwap]


def _trade_features(context, start, end):
    path = context.log_mid[start:end]
    step_volume = np.diff(context.volume[start:end])
    step_amount = np.diff(context.amount[start:end])
    amount = context.amount[end - 1] - context.amount[start]
    trades = context.trades[end - 1] - context.trades[start]
    volume = context.volume[end - 1] - context.volume[start]

    directional = step_volume > 0
    trade_price = np.full(len(step_volume), np.nan)
    trade_price[directional] = step_amount[directional] / step_volume[directional]
    side = np.sign(np.log(_positive(trade_price)) - path[:-1])
    side[~directional] = 0.0
    flow = side * np.where(directional, step_volume, 0.0)
    concentration = (len(step_volume) * np.sum((step_volume / volume) ** 2)
                     if volume != 0 else np.nan)
    return [
        amount,
        amount / trades if trades != 0 else np.nan,
        np.sum(flow) / volume if volume != 0 else np.nan,
        np.sum(np.diff(path) * flow) / np.sum(flow ** 2) if np.sum(flow ** 2) != 0 else np.nan,
        np.mean(step_volume == 0),
        concentration,
    ]


def _queue_churn(price, quantity):
    previous = quantity[:-1]
    holding = ((price[1:] == price[:-1]) & np.isfinite(price[1:]) &
               np.isfinite(previous) & np.isfinite(quantity[1:]) & (previous > 0))
    values = np.full(len(previous), np.nan)
    values[holding] = np.abs(quantity[1:][holding] - previous[holding]) / previous[holding]
    return _nanmean(values)


def _book_features(context, start, end, window_volume):
    block = slice(start, end)
    spread = context.book["spread"][block]
    depth = context.book["depth"][block]
    spread_mean = _nanmean(spread)
    depth_mean = _nanmean(depth)
    quote_valid = (np.isfinite(context.bid1[start:end]) & np.isfinite(context.ask1[start:end]))
    quote_pair = quote_valid[1:] & quote_valid[:-1]
    quote_changed = ((context.bid1[start + 1:end] != context.bid1[start:end - 1]) |
                     (context.ask1[start + 1:end] != context.ask1[start:end - 1]))
    quote_rate = quote_changed[quote_pair].mean() if quote_pair.any() else np.nan
    churn = _nanmean([
        _queue_churn(context.bid1[start:end], context.bid_qty[start:end, 0]),
        _queue_churn(context.ask1[start:end], context.ask_qty[start:end, 0]),
    ])
    return [
        spread_mean,
        _nanstd(spread) / spread_mean if spread_mean != 0 else np.nan,
        _nanmean(context.book["qi1"][block]),
        _nanmean(context.book["qi_gap"][block]),
        _nanmean(context.book["tqi"][block]),
        depth_mean / window_volume if window_volume != 0 else np.nan,
        _nanstd(depth) / depth_mean if depth_mean != 0 else np.nan,
        _nanmean(context.book["width"][block]),
        _nanmean(context.book["width_asym"][block]),
        _nanmean(context.book["far_press"][block]),
        quote_rate,
        churn,
        _nanmean(context.book["l1_count"][block]),
        _nanmean(context.book["l1_top"][block]),
    ]


def _day_features(context, end, day_start, day_end, bar_ticks):
    """日内状态：只用窗口末快照（end-1）及其之前的数据；bar 为行末 bar_ticks 条快照。"""
    tail = end - 1
    log_mid = context.log_mid[tail]
    bar_volume = context.volume[tail] - context.volume[tail - bar_ticks]
    elapsed_volume = context.volume[tail] - context.volume[day_start]
    average_volume = elapsed_volume * bar_ticks / (tail - day_start + 1.0)
    high, low = context.high_px[tail], context.low_px[tail]
    latest = context.lastpx[tail]
    range_position = (np.clip((latest - low) / (high - low), 0.0, 1.0)
                      if high != low else 0.5)
    return [
        log_mid - np.log(_positive(context.open_px[tail])),
        bar_volume / average_volume if average_volume != 0 else np.nan,
        np.log(_positive(context.max_px[tail])) - log_mid,
        log_mid - np.log(_positive(context.min_px[tail])),
        range_position,
        (tail - day_start) / (day_end - day_start - 1.0),
        float(context.afternoon[tail]),
        context.log_mid[tail - bar_ticks + 1] - context.log_mid[tail - bar_ticks],
    ]


def grid_counts(bid1, ask1, log_mid, width):
    """回放一张新开的固定半宽网格，返回（买次数, 卖次数, 残余敞口绝对值）。"""
    bid1 = _positive(np.asarray(bid1, dtype=np.float64))
    ask1 = _positive(np.asarray(ask1, dtype=np.float64))
    log_mid = np.asarray(log_mid, dtype=np.float64)
    if (len(bid1) < 1 or not np.isfinite(width) or width <= 0 or
            not np.isfinite(log_mid[0])):
        return np.array([np.nan, np.nan, np.nan])
    result = grid_engine.run_day(bid1, ask1, np.exp(log_mid),
                                 hard_exclude=None, width=width)
    buys, sells = result["buys"], result["sells"]
    return np.array([buys, sells, abs(buys - sells)], dtype=np.float64)


def _window_features(context, start, end, day_start, day_end, bar_ticks, width):
    path = context.log_mid[start:end]
    stats = path_stats(path)
    window_volume = context.volume[end - 1] - context.volume[start]
    columns = _price_features(context, start, end)
    columns += [stats[name][0] for name in PATH_NAMES]
    columns += _trade_features(context, start, end)
    columns += _book_features(context, start, end, window_volume)
    columns += _day_features(context, end, day_start, day_end, bar_ticks)
    columns += grid_counts(context.bid1[start:end], context.ask1[start:end],
                           path, width).tolist()
    return np.asarray(columns, dtype=np.float64)


def _forward_targets(log_mid, invalid, lookback, pred_ticks):
    """当日逐 tick 的前瞻目标 (N, 5)：行 t 的路径为 [t, t+H]。

    路径统计量对全部滑窗一次算出；回看窗口不完整（t<L−1）、前瞻越界或路径含无效
    快照的行记 NaN，有效行范围与特征一致。
    """
    n = len(log_mid)
    targets = np.full((n, len(TARGET_NAMES)), np.nan)
    rows = n - pred_ticks
    if rows <= 0:
        return targets
    stats = path_stats(sliding_window_view(log_mid, pred_ticks + 1))
    prefix = np.concatenate([[0], np.cumsum(invalid)])
    usable = (prefix[pred_ticks + 1:] - prefix[:rows]) == 0
    usable[:min(lookback - 1, rows)] = False
    targets[:rows] = np.where(
        usable[:, None], np.column_stack([stats[name] for name in TARGET_NAMES]), np.nan)
    return targets


def build_symbol(symbol: str, data_dir: str = "data",
                 spec: WindowSpec = WindowSpec()) -> dict:
    """构建一个标的的统一缓存内容（行索引即 tick 索引）。

    返回 CACHE_ARRAYS 的全部数组：preds 置 NaN 待 forecast.train 回写，R 为该标的单日
    最大快照数，窗口不完整、前瞻越界与尾部对齐的行均记 NaN。半宽取 strategy/width.py
    的逐日 grid_width，历史不足的日期 width 记 nan，对应网格成交特征为 nan。
    """
    days = load_days(symbol, data_dir=data_dir, atr_days=spec.atr_window)
    if not days:
        raise ValueError(f"标的 {symbol} 没有连续竞价快照")
    frame = pd.concat([d.frame for d in days], ignore_index=True)
    frame["l1_top"] = top_order_share(frame)

    dates, day = np.unique(frame["MDDate"].astype(str).to_numpy(), return_inverse=True)
    counts = np.bincount(day, minlength=len(dates))
    day_offsets = np.concatenate([[0], np.cumsum(counts)])
    context = _WindowContext(frame)

    # 行有效性：输入窗口内中间价路径与累计成交字段均有限（向量化前缀和判定）
    invalid = ~(np.isfinite(context.log_mid) & np.isfinite(context.volume)
                & np.isfinite(context.amount) & np.isfinite(context.trades))
    invalid_cumsum = np.concatenate([[0], np.cumsum(invalid)])

    rows = int(counts.max())
    width_by_date = grid_width(daily_bars(frame), atr_mult=spec.atr_mult,
                               atr_window=spec.atr_window,
                               min_width_ratio=spec.min_width_ratio)
    width_by_date.index = width_by_date.index.astype(str)
    widths = np.array([width_by_date.get(date, np.nan) for date in dates], dtype=np.float64)

    features = np.full((len(dates), rows, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    targets = np.full((len(dates), rows, len(TARGET_NAMES)), np.nan, dtype=np.float32)
    for date_index in range(len(dates)):
        day_start, day_end = day_offsets[date_index], day_offsets[date_index + 1]
        day_ticks = day_end - day_start
        targets[date_index, :day_ticks] = _forward_targets(
            context.log_mid[day_start:day_end], invalid[day_start:day_end],
            spec.lookback_ticks, spec.pred_ticks)
        for t in range(spec.lookback_ticks - 1, day_ticks):
            end = day_start + t + 1          # 窗口右开端，末快照即 tick t
            start = end - spec.lookback_ticks
            if invalid_cumsum[end] - invalid_cumsum[start] == 0:
                features[date_index, t] = _window_features(
                    context, start, end, day_start, day_end,
                    spec.bar_ticks, widths[date_index])

    return {"dates": dates.astype("U8"), "features": features, "targets": targets,
            "width": widths.astype(np.float32),
            "preds": np.full_like(targets, np.nan)}


def cache_path(symbol: str, cache_dir: str = "cache") -> str:
    """标的统一缓存的路径：cache/<symbol>.npz。"""
    return os.path.join(cache_dir, f"{symbol}.npz")


def _metadata(symbol, data_dir, spec: WindowSpec):
    return {
        "schema": CACHE_SCHEMA,
        "symbol": symbol,
        "source": symbol_source_signature(data_dir, symbol),
        "window": asdict(spec),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "prediction": None,   # 训练后由 write_predictions 回写 {"data_hash": ...}
    }


def load_cache(symbol: str, data_dir: str = "data", cache_dir: str = "cache",
               spec: WindowSpec = WindowSpec(), zero_nan: bool = True) -> dict:
    """加载标的统一缓存（缺失或窗口级 metadata 不匹配时重建）。

    返回 {"dates": (D,), "features": (D,R,47), "targets": (D,R,5), "preds": (D,R,5),
    "width": (D,)}，R 为该标的单日最大快照数、行索引即 tick 索引；preds 在
    forecast.train 回写前全为 NaN。zero_nan=False 保留 NaN（LightGBM 原生处理缺失），
    True 时置 0（进控制器状态：神经网络不接受 NaN）。
    """
    path = cache_path(symbol, cache_dir)
    meta = _metadata(symbol, data_dir, spec)
    window_keys = [key for key in meta if key != "prediction"]
    rebuild = True
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            stored = json.loads(str(z["_metadata"]))
        rebuild = {key: stored.get(key) for key in window_keys} != {
            key: meta[key] for key in window_keys}
    if rebuild:
        entry = build_symbol(symbol, data_dir=data_dir, spec=spec)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, _metadata=json.dumps(meta, sort_keys=True), **entry)
    with np.load(path, allow_pickle=False) as z:
        entry = {name: z[name] for name in CACHE_ARRAYS}
    if zero_nan:
        for name in ("features", "targets", "preds"):
            entry[name] = np.nan_to_num(entry[name])
    return entry


def write_predictions(symbol: str, preds, data_hash: str, cache_dir: str = "cache") -> None:
    """把逐样本行预测回写进统一缓存的 preds 块，并记录训练时的 data_hash。"""
    path = cache_path(symbol, cache_dir)
    with np.load(path, allow_pickle=False) as z:
        entry = {name: z[name] for name in CACHE_ARRAYS if name != "preds"}
        metadata = json.loads(str(z["_metadata"]))
    metadata["prediction"] = {"data_hash": data_hash}
    np.savez(path, _metadata=json.dumps(metadata, sort_keys=True),
             **entry, preds=np.asarray(preds, dtype=np.float32))


def predictions_current(symbol: str, data_hash: str, cache_dir: str = "cache") -> bool:
    """统一缓存的预测块是否由给定 data_hash 训练得到（缺失或不一致则需重训回写）。"""
    path = cache_path(symbol, cache_dir)
    if not os.path.exists(path):
        return False
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["_metadata"]))
    return (metadata["prediction"] or {}).get("data_hash") == data_hash
