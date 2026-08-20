"""标的数据装配：统一缓存 + 报价序列 → 逐锚点实现路径统计与回放输入。

每标的装配一个 bank，是模式定义、识别训练、经济验证与统一回测的共同数据源
（行索引即压缩分钟索引，M=237）：
- realized (D,M,5)：实现的 5 维前瞻路径统计，resid_abs_q90 / abs_slope 以当日
  相对半宽 w = 半宽/锚点 mid 归一，其余 3 维保持原值；
- judgeable (D,M)：5 维全有限 ∧ w 有效，模式判定只在该掩码内进行；
- features (D,M,47)：识别器输入（NaN 由 LightGBM 原生处理）；
- anchors (D,M)：各分钟末快照的当日 tick 索引（无快照记 -1）；
- quotes {日索引: (bid1, ask1, mid)} 与 open_px {日索引: 当日开盘价}；
- split：该标的的时序切分（逐标的切分，异日历标的不互相泄漏）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_provider.split import chronological_split
from data_provider.ticks import load_days
from data_provider.windows import TARGET_NAMES, load_cache
from strategy import costs, engine, metrics

from forecast.regime.config import RegimeConfig

RESID_IDX = TARGET_NAMES.index("resid_abs_q90")
SLOPE_IDX = TARGET_NAMES.index("abs_slope")


@dataclass
class SymbolBank:
    symbol: str
    dates: np.ndarray          # (D,) 日期字符串
    split: object              # chronological_split 的返回
    width: np.ndarray          # (D,) 当日固定半宽
    features: np.ndarray       # (D,M,47)
    realized: np.ndarray       # (D,M,5) 归一后的实现路径统计
    judgeable: np.ndarray      # (D,M) bool
    anchors: np.ndarray        # (D,M) 分钟锚点的当日 tick 索引，无快照记 -1
    quotes: dict               # {日索引: (bid1, ask1, mid)}，仅有报价数据的日
    open_px: dict              # {日索引: 当日开盘价}

    def day_indices(self, flag: str) -> list[int]:
        """某切分段内可回放的日索引：有报价、半宽有效、存在可判定锚点。"""
        wanted = set(getattr(self.split, flag))
        return [i for i, date in enumerate(self.dates)
                if date in wanted and i in self.quotes
                and np.isfinite(self.width[i]) and self.judgeable[i].any()]

    def replay_start(self, day_index: int, start_minute: int) -> int:
        """回放起点 tick：分钟 start_minute 起的首个锚点（该分钟无快照时取其后）。"""
        anchors = self.anchors[day_index, start_minute:]
        valid = anchors[anchors >= 0]
        return int(valid[0]) if len(valid) else len(self.quotes[day_index][2])


def expand_minutes(values: np.ndarray, anchors: np.ndarray, n_ticks: int) -> np.ndarray:
    """把逐分钟值按锚点前向填充到逐 tick：tick t 取最近一个锚点 ≤ t 的值。

    首锚点之前记 NaN（因果性：分钟值在锚点即该分钟末快照处才可得）。
    """
    out = np.full(n_ticks, np.nan)
    valid = anchors >= 0
    out[anchors[valid]] = np.asarray(values, dtype=np.float64)[valid]
    latest = np.maximum.accumulate(
        np.where(np.isfinite(out), np.arange(n_ticks), -1))
    return np.where(latest >= 0, out[np.maximum(latest, 0)], np.nan)


def load_bank(symbol: str, cfg: RegimeConfig) -> SymbolBank:
    cache = load_cache(symbol, data_dir=cfg.data_dir, cache_dir=cfg.cache_dir,
                       spec=cfg.window, zero_nan=False)
    dates = cache["dates"]
    anchors = cache["anchor_ticks"]
    days = {day.date: day for day in load_days(symbol, data_dir=cfg.data_dir,
                                               atr_days=cfg.window.atr_window)}

    quotes, open_px = {}, {}
    anchor_mid = np.full(anchors.shape, np.nan)
    for i, date in enumerate(dates):
        if date not in days:
            continue
        frame = days[date].frame
        bid1 = frame["Buy1Price"].to_numpy(np.float64)
        ask1 = frame["Sell1Price"].to_numpy(np.float64)
        mid = 0.5 * (bid1 + ask1)
        quotes[i] = (bid1, ask1, mid)
        open_px[i] = float(days[date].open_px)
        valid = anchors[i] >= 0
        anchor_mid[i, valid] = mid[anchors[i, valid]]

    width = np.asarray(cache["width"], dtype=np.float64)
    valid_width = np.isfinite(width) & (width > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = (np.where(valid_width, width, np.nan)[:, None]
                    / np.where(anchor_mid > 0, anchor_mid, np.nan))
        realized = np.array(cache["targets"], dtype=np.float64)
        for idx in (RESID_IDX, SLOPE_IDX):
            realized[..., idx] = realized[..., idx] / relative
    judgeable = np.isfinite(realized).all(axis=-1)

    return SymbolBank(
        symbol=symbol, dates=dates, split=chronological_split(list(dates)),
        width=width, features=cache["features"], realized=realized,
        judgeable=judgeable, anchors=anchors, quotes=quotes, open_px=open_px)


def load_banks(cfg: RegimeConfig) -> dict[str, SymbolBank]:
    banks = {}
    for symbol in sorted(cfg.symbols):
        banks[symbol] = load_bank(symbol, cfg)
        print(f"bank {symbol}: {len(banks[symbol].quotes)} 日", flush=True)
    return banks


def replay_grid_day(bank: SymbolBank, day_index: int, mask_full, cfg: RegimeConfig,
                    start_minute: int | None = None) -> dict:
    """单日 engine 回放（mask_full=None 为常开），返回 metrics.summarize 的日记录。

    口径同统一回测：自可预测起点（缺省分钟 lookback_min−1，回看窗满的首个锚点）起、
    固定半宽 W_d、锚点门控节奏加连续确认（confirm_n），锚点中心重建同 engine 规则；
    g = 费用后净利润 / W_d。回放区间为空时返回 {}。
    """
    bid1, ask1, mid = bank.quotes[day_index]
    if start_minute is None:
        start_minute = cfg.window.lookback_min - 1
    t0 = bank.replay_start(day_index, start_minute)
    if t0 >= len(mid):
        return {}
    anchors = bank.anchors[day_index]
    anchors = anchors[anchors >= t0] - t0
    width = float(bank.width[day_index])
    result = engine.run_day(
        bid1[t0:], ask1[t0:], mid[t0:],
        hard_exclude=None if mask_full is None else mask_full[t0:],
        width=width, anchors=anchors, confirm_n=cfg.confirm_n, trace=True)
    fills = costs.fills_from_events(result["events"])
    return {"g": costs.daily_net(fills, float(mid[-1])) / width,
            "n_buys": result["buys"], "n_sells": result["sells"],
            "closure_rate": metrics.closure_rate(result["buys"], result["sells"]),
            "width_rel": width / bank.open_px[day_index]}
