"""标的数据装配：统一缓存 + 报价序列 → 逐 tick 实现路径统计与回放输入。

每标的装配一个 bank，是模式定义、识别训练、经济验证与统一回测的共同数据源：
- realized (D,R,5)：实现的 5 维前瞻路径统计，resid_abs_q90 / abs_slope 以当日
  相对半宽 w = 半宽/当拍 mid 归一，其余 3 维保持原值；
- judgeable (D,R)：5 维全有限 ∧ w 有效，模式判定只在该掩码内进行；
- features (D,R,47)：识别器输入（NaN 由 LightGBM 原生处理）；
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
    features: np.ndarray       # (D,R,47)
    realized: np.ndarray       # (D,R,5) 归一后的实现路径统计
    judgeable: np.ndarray      # (D,R) bool
    quotes: dict               # {日索引: (bid1, ask1, mid)}，仅有报价数据的日
    open_px: dict              # {日索引: 当日开盘价}

    def day_indices(self, flag: str) -> list[int]:
        """某切分段内可回放的日索引：有报价、半宽有效、存在可判定 tick。"""
        wanted = set(getattr(self.split, flag))
        return [i for i, date in enumerate(self.dates)
                if date in wanted and i in self.quotes
                and np.isfinite(self.width[i]) and self.judgeable[i].any()]


def load_bank(symbol: str, cfg: RegimeConfig) -> SymbolBank:
    cache = load_cache(symbol, data_dir=cfg.data_dir, cache_dir=cfg.cache_dir,
                       spec=cfg.window, zero_nan=False)
    dates = cache["dates"]
    n_days, n_ticks = cache["targets"].shape[:2]
    days = {day.date: day for day in load_days(symbol, data_dir=cfg.data_dir,
                                               atr_days=cfg.window.atr_window)}

    quotes, open_px = {}, {}
    anchors = np.full((n_days, n_ticks), np.nan)
    for i, date in enumerate(dates):
        if date not in days:
            continue
        frame = days[date].frame
        bid1 = frame["Buy1Price"].to_numpy(np.float64)
        ask1 = frame["Sell1Price"].to_numpy(np.float64)
        mid = 0.5 * (bid1 + ask1)
        quotes[i] = (bid1, ask1, mid)
        open_px[i] = float(days[date].open_px)
        anchors[i, :len(mid)] = mid

    width = np.asarray(cache["width"], dtype=np.float64)
    valid_width = np.isfinite(width) & (width > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = (np.where(valid_width, width, np.nan)[:, None]
                    / np.where(anchors > 0, anchors, np.nan))
        realized = np.array(cache["targets"], dtype=np.float64)
        for idx in (RESID_IDX, SLOPE_IDX):
            realized[..., idx] = realized[..., idx] / relative
    judgeable = np.isfinite(realized).all(axis=-1)

    return SymbolBank(
        symbol=symbol, dates=dates, split=chronological_split(list(dates)),
        width=width, features=cache["features"], realized=realized,
        judgeable=judgeable, quotes=quotes, open_px=open_px)


def load_banks(cfg: RegimeConfig) -> dict[str, SymbolBank]:
    banks = {}
    for symbol in sorted(cfg.symbols):
        banks[symbol] = load_bank(symbol, cfg)
        print(f"bank {symbol}: {len(banks[symbol].quotes)} 日", flush=True)
    return banks


def replay_grid_day(bank: SymbolBank, day_index: int, mask_full, cfg: RegimeConfig,
                    t0_tick: int | None = None) -> dict:
    """单日 engine 回放（mask_full=None 为常开），返回 metrics.summarize 的日记录。

    口径同统一回测：自可预测起点 t0（缺省 lookback_ticks，回看窗满）起、固定半宽 W_d、
    每 stride_ticks 判定一次；g = 费用后净利润 / W_d。回放区间为空时返回 {}。
    """
    bid1, ask1, mid = bank.quotes[day_index]
    t0 = cfg.window.lookback_ticks if t0_tick is None else t0_tick
    take = np.arange(t0, len(mid))
    if len(take) == 0:
        return {}
    width = float(bank.width[day_index])
    result = engine.run_day(
        bid1[take], ask1[take], mid[take],
        hard_exclude=None if mask_full is None else mask_full[take],
        width=width, decide_interval=cfg.stride_ticks, trace=True)
    fills = costs.fills_from_events(result["events"])
    return {"g": costs.daily_net(fills, float(mid[take][-1])) / width,
            "n_buys": result["buys"], "n_sells": result["sells"],
            "closure_rate": metrics.closure_rate(result["buys"], result["sells"]),
            "width_rel": width / bank.open_px[day_index]}
