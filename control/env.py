"""交易环境：定长决策网格的逐区间 rollout、撮合模拟、奖励与日内统计。

design.md 第 3–7 节的实现：
  - 决策点为固定每 decision_interval_min 分钟的锚点加日终（4.1），成交不触发决策；
    动作 (h, q) 给出对称的两条触发线与每笔成交手数，区间内逐笔扫描——每笔成交后
    中心重置为成交价、重算边界与可成交量（仓位 / 现金约束逐笔重判）并继续扫至区间末；
  - h = 0 时在决策点按对手方一档价平回底仓（3.5，仅净持仓非零时可选），
    h 极大（关闭档）则网格无法触发；
  - 中心重置规则（3.5）：每笔成交后中心即成交价；区间无成交则决策锚点处中心改用
    当拍中间价；
  - 被动单以对手方一档严格穿越边界为成交条件，成交价为取整后的边界价（3.2）；
  - 区间奖励按权益差口径扣除底仓被动收益（6.1），训练奖励另加 hindsight bonus 与
    存货惩罚，两者的仓位取区间内时间加权（6.2）。

网格几何（半宽、边界取整、严格穿越判定）与 strategy/grid.py 共用同一定义，
显性费用一律走 strategy/costs.py。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from data_provider.ticks import (MINUTES_PER_DAY, DayData, anchor_ffill,
                                 minute_anchors, minute_index)
from strategy.costs import COMMISSION_RATE, fee
from strategy.grid import boundaries, buy_crossed, half_width, sell_crossed
from strategy.metrics import closure_rate

from .config import Config
from .features import (
    PRIV_CASH,
    PRIV_CENTER,
    PRIV_CUM_BUYS,
    PRIV_CUM_SELLS,
    PRIV_DEC_MID,
    PRIV_INT_LOTS,
    PRIV_INT_NOTIONAL,
    PRIV_LAST_FILL,
    PRIV_LAST_PX,
    PRIV_POS,
    PRIV_RAW_DIM,
    PRIV_SIZE,
    PRIV_WIDTH,
    PRIVATE_DIM,
    PRED_DIM,
    WINDOW_DIM,
    FeatureStats,
    build_macro_features,
    build_micro_matrix,
)

RANGE_TO_SIGMA = math.sqrt(8.0 / math.pi)  # 布朗运动 E[极差] = √(8/π)·σ ≈ 1.6σ


@dataclass
class Observation:
    micro_lob: np.ndarray   # (micro_steps, MICRO_DIM)
    private: np.ndarray     # (micro_steps, PRIVATE_DIM)
    macro: np.ndarray       # (MACRO_DIM,)
    symbol_id: int = 0      # 标的在 cfg.symbols 中的索引（网络端 embedding）
    flatten_allowed: bool = True   # 平仓档是否可选：净持仓（pos − Q0）非零（design 5.1）


class DayMarket:
    """单个交易日的预计算市场数据与观测构建器。

    window 为该日的样本行块（M×29：24 维窗口统计 + 5 维前瞻预测，行索引即分钟索引，
    见 data/features.md §3 与 §5.2）；缺省为零块（无缓存时的降级形态，如冒烟测试的
    合成数据）。分钟网格（决策锚点、微观 / 私有序列抽样与 1 分钟 bar）统一取自
    data_provider/ticks.py 的 minute_anchors。
    """

    def __init__(self, day: DayData, cfg: Config, window: np.ndarray | None = None,
                 symbol_id: int = 0):
        f = day.frame
        self.cfg = cfg
        self.date = day.date
        self.atr = day.atr
        self.pre_close = day.pre_close
        self.open_px = day.open_px
        self.bid1 = f["Buy1Price"].to_numpy(np.float64)
        self.ask1 = f["Sell1Price"].to_numpy(np.float64)
        # 一档挂单量由股折算为手，供 liquidity_ratio 监控成交规模（design 3.2）
        self.bid1_lots = f["Buy1OrderQty"].to_numpy(np.float64) / cfg.lot_size
        self.ask1_lots = f["Sell1OrderQty"].to_numpy(np.float64) / cfg.lot_size
        self.mid = (self.bid1 + self.ask1) / 2.0
        tvt = f["TotalVolumeTrade"].to_numpy(np.float64)
        self.vol_delta = np.maximum(np.diff(tvt, prepend=tvt[0]), 0.0)
        self.micro = build_micro_matrix(f)
        self.n = len(self.mid)

        # 分钟网格：逐 tick 分钟索引、锚点与前向填充锚点（决策 / 抽样 / bar 共用）
        self.minute = minute_index(f["MDTime"])
        self.anchors = minute_anchors(f)
        self.anchor_fill = anchor_ffill(self.anchors)
        self.bars = self._minute_bars()

        # 参与扫描的快照：盘口未交叉，且累计成交笔数刷新历史高点。
        # fmax 累积使计数缺失或回退的快照不判成交，也不改变比较基准。
        num_trades = f["NumTrades"].to_numpy(np.float64)
        self.valid_book = self.bid1 <= self.ask1
        self.scannable = np.empty(self.n, dtype=bool)
        self.scannable[0] = False
        self.scannable[1:] = (
            self.valid_book[1:]
            & (num_trades[1:] > np.fmax.accumulate(num_trades)[:-1])
        )
        # 拟合标准化统计量所用的固定分钟锚点采样（与决策相位无关）
        self.sample_points = [int(a) for a in self.anchors[cfg.lookback_min - 1:]
                              if a >= 0]
        self.stats: FeatureStats | None = None
        self.symbol_id = symbol_id
        self.window = window if window is not None else np.zeros(
            (MINUTES_PER_DAY, WINDOW_DIM + PRED_DIM), dtype=np.float32
        )

    def _minute_bars(self) -> np.ndarray:
        """逐分钟 bar (M, 5)：中间价 [open, high, low, close] 与成交量。

        无快照的分钟以最近已有 close 前向填充（日初缺失取当日首个 close）、量能记 0，
        使回看 bar 窗口对缺失分钟稳定。
        """
        bars = np.full((MINUTES_PER_DAY, 5), np.nan)
        starts = np.concatenate([[0], np.flatnonzero(np.diff(self.minute)) + 1])
        rows = self.minute[starts]
        ends = np.concatenate([starts[1:], [self.n]])
        bars[rows, 0] = self.mid[starts]
        bars[rows, 1] = np.maximum.reduceat(self.mid, starts)
        bars[rows, 2] = np.minimum.reduceat(self.mid, starts)
        bars[rows, 3] = self.mid[ends - 1]
        bars[rows, 4] = np.add.reduceat(self.vol_delta, starts)
        close = bars[:, 3]
        latest = np.maximum.accumulate(
            np.where(np.isfinite(close), np.arange(MINUTES_PER_DAY), -1))
        first = close[np.isfinite(close)][0]
        filled = np.where(latest >= 0, close[np.maximum(latest, 0)], first)
        missing = ~np.isfinite(close)
        for column in range(4):
            bars[missing, column] = filled[missing]
        bars[missing, 4] = 0.0
        return bars

    def decision_tick(self, minute: int) -> int:
        """分钟 minute 的决策 tick：该分钟末快照，无快照时前向填充到最近锚点。"""
        return int(self.anchor_fill[min(minute, MINUTES_PER_DAY - 1)])

    @property
    def start(self) -> int:
        """缺省回合起点（无偏移）：回看窗满首分钟（lookback_min−1）的锚点。"""
        return self.decision_tick(self.cfg.lookback_min - 1)

    @property
    def tradable(self) -> bool:
        """需有有效 ATR（前 A 日完整）与至少一个可推进的决策区间。"""
        if not np.isfinite(self.atr) or self.atr <= 0.0:
            return False
        return (self.anchors[self.cfg.lookback_min - 1:] >= 0).any() and self.start < self.n - 1

    @property
    def p0(self) -> float:
        """缺省起点的建仓参考价（评估口径；训练回合的 p0 由 TradingEnv 按起点定义）。"""
        return float(self.mid[self.start])

    @property
    def equity0(self) -> float:
        """缺省起点的初始权益 E0 = 2Q0·p0（观测归一用）。"""
        return self.cfg.max_position * self.p0

    @property
    def base_value(self) -> float:
        """缺省起点的归一基准 B = Q0·p0（design 3.5；评估与回测的 g 分母）。"""
        return self.cfg.base_position * self.p0

    @property
    def sigma_d(self) -> float:
        """当日收益率标准差的估计 σ_d = ATR3 / (1.6·前收)（design 6.2）。"""
        return self.atr / (RANGE_TO_SIGMA * self.pre_close)

    def set_stats(self, stats: FeatureStats | None) -> None:
        """挂载标准化统计量（本标的行由 symbol_id 索引）并就地变换微观矩阵（仅调用一次）。"""
        if stats is not None:
            self.micro = stats.micro(self.micro, self.symbol_id)
        self.stats = stats

    def sample_times(self, t: int) -> np.ndarray:
        """决策点 t 的回看抽样时刻：末 micro_steps 个分钟的锚点，末项恰为 t。

        无快照的分钟前向填充到最近锚点（重复前帧），早于当日首个锚点的分钟取
        首个锚点，与微观、私有序列逐行对应。
        """
        m = self.minute[t]
        times = self.anchor_fill[m - self.cfg.micro_steps + 1: m + 1]
        first = self.anchors[self.anchors >= 0][0]
        return np.where(times >= 0, times, first)

    def micro_window(self, t: int) -> np.ndarray:
        """回看窗口内按分钟锚点抽样的微观序列 (micro_steps, MICRO_DIM)。

        取每分钟的末快照，故序列末帧恰为决策点 t 的快照，与宏观 bar 的收盘价
        对齐——决策所依据的最新盘口与撮合所用的盘口为同一快照。
        """
        return self.micro[self.sample_times(t)]

    def macro_at(self, t: int, normalized: bool = True) -> np.ndarray:
        cfg = self.cfg
        m = self.minute[t]
        macro = build_macro_features(self.bars[m - cfg.n_bars + 1: m + 1])
        # 行索引即分钟索引：第 m 行的窗口恰好收于分钟 m 的锚点，不含未完结数据
        macro = np.concatenate([macro, self.window[m]])
        if normalized and self.stats is not None:
            macro = self.stats.macro(macro, self.symbol_id)
        return macro

    def observe(self, t: int, priv_hist: np.ndarray) -> Observation:
        """构建决策点 t 的观测。

        priv_hist 为 (micro_steps+1, PRIV_RAW_DIM) 的私有状态原始记录：末 micro_steps
        行与 sample_times(t) 一一对应（末行即 t 的当期状态），首行为窗口前一分钟的
        锚点状态，供累计成交笔数差分出首步的步内笔数。
        """
        cfg = self.cfg
        times = self.sample_times(t)
        rows = priv_hist[1:]
        private = np.empty((cfg.micro_steps, PRIVATE_DIM), dtype=np.float32)
        private[:, 0] = rows[:, PRIV_POS] / cfg.max_position
        private[:, 1] = rows[:, PRIV_CASH] / self.equity0
        private[:, 2] = (self.n - 1 - times) / self.n
        private[:, 3] = (self.mid[times] - rows[:, PRIV_CENTER]) / self.atr
        private[:, 4] = rows[:, PRIV_WIDTH] / cfg.widths[-1]
        private[:, 5] = rows[:, PRIV_SIZE] / cfg.sizes[-1]
        private[:, 6] = (times - rows[:, PRIV_LAST_FILL]) / self.n
        private[:, 7] = np.log1p(np.maximum(np.diff(priv_hist[:, PRIV_CUM_BUYS]), 0.0))
        private[:, 8] = np.log1p(np.maximum(np.diff(priv_hist[:, PRIV_CUM_SELLS]), 0.0))
        last_px = rows[:, PRIV_LAST_PX]
        private[:, 9] = np.where(
            last_px > 0.0, (last_px - rows[:, PRIV_DEC_MID]) / self.atr, 0.0)
        lots = rows[:, PRIV_INT_LOTS]
        private[:, 10] = np.where(
            lots > 0.0,
            rows[:, PRIV_INT_NOTIONAL] / np.where(lots > 0.0, lots, 1.0) / self.atr,
            0.0)
        return Observation(
            micro_lob=self.micro_window(t),
            private=private,
            macro=self.macro_at(t),
            symbol_id=self.symbol_id,
            flatten_allowed=bool(rows[-1, PRIV_POS] != cfg.base_position),
        )

    def trade(self, t: int, qty: float, price: float) -> tuple[float, float, float]:
        """在快照 t 处按给定价格成交 qty 手（正买负卖）。

        返回（现金变动, 手续费, 执行成本）：手续费为显性费用，执行成本为成交价相对
        中间价的偏离。两者之和即 3.3 的交易成本，与现金账户记账完全一致。
        """
        side = 1.0 if qty > 0 else -1.0
        notional = abs(qty) * price
        fee_cost = fee(side, notional)
        return -side * notional - fee_cost, fee_cost, side * (notional - abs(qty) * self.mid[t])

    def hindsight_price(self, t: int) -> float:
        """hindsight 标签的未来价：pred_min 分钟后的锚点中间价（不跨日，尾部截断）。"""
        horizon = min(self.minute[t] + self.cfg.window.pred_min, MINUTES_PER_DAY - 1)
        return float(self.mid[max(self.anchor_fill[horizon], t)])


@dataclass(frozen=True)
class GridParams:
    """规则层参数：一次决策所设定的网格状态（design 第 3 节）。"""

    width: float        # 对称半宽（× ATR3），0 表示平回底仓，不必落在动作梯子上
    size: int           # 每次触发成交手数


def action_params(cfg: Config, action: tuple[int, int]) -> GridParams:
    """把动作档位索引（半宽档, 数量档）翻译为规则层参数。"""
    return GridParams(cfg.widths[action[0]], cfg.sizes[action[1]])


@dataclass
class Fill:
    tick: int
    qty: float          # 带符号成交手数（正买负卖）
    price: float
    liquidity: float    # 成交快照的对手一档深度（手）
    center_move: float  # 成交价相对上一中心的移动距离（bp）
    kind: str           # "grid" 区间内触发 / "immediate" 决策点立即成交 /
                        # "flatten" 决策点平仓 / "liquidate" 日终清仓（3.5）


@dataclass
class Interval:
    tau: int               # 区间时长（tick，诊断的时间加权用）
    excess: float          # 区间内时间加权平均超额敞口（手，带符号）
    abs_excess: float      # 时间加权平均绝对超额敞口（手）
    max_abs_excess: float  # 区间内最大绝对超额敞口（手）
    boundary_share: float  # |excess| 触及仓位带边界（≥ Q0）的时间占比
    width: float           # 生效半宽档（× ATR3）
    size: int              # 生效成交手数（平仓时记 0）
    inventory_load: float  # (时间加权平均 I/B)^2


@dataclass
class StepResult:
    obs: Observation | None
    reward: float            # 超额 P&L 奖励（评估口径）
    train_reward: float      # 按 σ_d 归一、含 hindsight bonus 与存货惩罚的训练奖励
    done: bool
    t: int                   # 下一决策点的 tick 索引
    priv_hist: np.ndarray    # 下一决策点的私有状态历史


class TradingEnv:
    """单交易日的网格环境；开盘持底仓 Q0，日终只平回净敞口。

    start_offset_min 为回合起点相对窗满首分钟的偏移（design 7.1 的训练起点随机化，
    决策网格整体平移）；验证 / 测试一律取 0。p0 与奖励基准 B = Q0·p0 逐回合按起点定义。
    """

    def __init__(self, market: DayMarket, hindsight: bool = True,
                 start_offset_min: int = 0):
        self.market = market
        self.cfg = market.cfg   # 与市场同源，避免两份配置静默分歧
        self.hindsight = hindsight
        self.minute = self.cfg.lookback_min - 1 + start_offset_min
        self.t = market.decision_tick(self.minute)
        self.p0 = float(market.mid[self.t])
        self.base_value = self.cfg.base_position * self.p0   # B = Q0·p0（design 3.5）
        self.pos = float(self.cfg.base_position)
        self.cash = self.base_value        # 建立底仓后的剩余现金 Q0·p0
        self.center = self.p0              # 首个决策点以中间价建网，该快照只建网不成交
        self.last_fill_tick = self.t
        self.n_steps = 0
        self.fee_cost = 0.0        # 当日累计手续费（显性）
        self.execution_cost = 0.0  # 当日累计成交价相对中间价的偏离
        self.fills: list[Fill] = []
        self.intervals: list[Interval] = []
        # 区间成交过程的原始量（features §5.3 新增 4 通道的来源）
        self.cum_buys = self.cum_sells = 0
        self.dec_mid = self.last_px = 0.0
        self.int_notional = self.int_lots = 0.0
        # 私有状态按 tick 记录：回看窗口按分钟锚点抽样，与决策相位解耦
        self.priv_raw = np.zeros((market.n, PRIV_RAW_DIM), dtype=np.float32)
        self._write_priv(0, self.t, width=0.0, size=0)
        # 建网前的填充段不携带网格状态：中心偏离与距上次成交时长均记 0
        self.priv_raw[: self.t + 1, PRIV_CENTER] = market.mid[: self.t + 1]
        self.priv_raw[: self.t + 1, PRIV_LAST_FILL] = np.arange(self.t + 1)

    # ---- 私有状态 ----
    def _write_priv(self, lo: int, hi: int, width: float, size: int) -> None:
        """将当前账户、网格与区间成交状态写入 tick 区间 [lo, hi]。"""
        rows = self.priv_raw[lo : hi + 1]
        rows[:, PRIV_POS] = self.pos
        rows[:, PRIV_CASH] = self.cash
        rows[:, PRIV_CENTER] = self.center
        rows[:, PRIV_WIDTH] = width
        rows[:, PRIV_SIZE] = size
        rows[:, PRIV_LAST_FILL] = self.last_fill_tick
        rows[:, PRIV_CUM_BUYS] = self.cum_buys
        rows[:, PRIV_CUM_SELLS] = self.cum_sells
        rows[:, PRIV_DEC_MID] = self.dec_mid
        rows[:, PRIV_LAST_PX] = self.last_px
        rows[:, PRIV_INT_NOTIONAL] = self.int_notional
        rows[:, PRIV_INT_LOTS] = self.int_lots

    def priv_window(self, t: int) -> np.ndarray:
        """决策点 t 的私有状态历史 (micro_steps+1, PRIV_RAW_DIM)。

        首行为窗口前一分钟的锚点（日初缺失取 tick 0，累计量为 0），
        供 observe 差分出首步的步内成交笔数。
        """
        m = self.market
        minute = m.minute[t]
        lead_minute = minute - self.cfg.micro_steps
        lead = m.anchor_fill[lead_minute] if lead_minute >= 0 else -1
        times = np.concatenate([[max(int(lead), 0)], m.sample_times(t)])
        return self.priv_raw[times].copy()

    def observation(self) -> Observation:
        return self.market.observe(self.t, self.priv_window(self.t))

    # ---- 撮合 ----
    def _lines(self, width: float) -> tuple[float, float]:
        """当前中心下对称的两条触发线（卖出线, 买入线）。

        半宽为 ATR3 倍数（下限为前收的 min_width_ratio，与逐日固定半宽同一参考价，
        见 features.md §2）；卖出线向上取整、买入线向下取整到最小变动价位，
        保证是合法限价且不缩窄名义半宽（几何定义见 strategy/grid.py）。
        """
        hw = half_width(width, self.market.atr, self.market.pre_close,
                        self.cfg.window.min_width_ratio)
        return boundaries(self.center, hw, self.cfg.tick_size)

    def _sell_lots(self, size: int) -> float:
        """卖出侧的有效成交量：受当前持仓截断。"""
        return min(float(size), self.pos)

    def _buy_lots(self, size: int, price: float) -> float:
        """买入侧的有效成交量：受仓位上界与现金约束。"""
        cfg = self.cfg
        lots = min(float(size), cfg.max_position - self.pos)
        if lots <= 0.0:
            return 0.0
        if self.cash < lots * price * (1.0 + COMMISSION_RATE):
            return 0.0
        return lots

    def _execute(self, t: int, qty: float, price: float, kind: str) -> None:
        """按给定价格成交 qty 手、登记成交并记账。

        成交后中心重置到成交价（3.5），并更新区间成交统计（features §5.3）。
        """
        cash_delta, fee_cost, execution = self.market.trade(t, qty, price)
        self.fee_cost += fee_cost
        self.execution_cost += execution
        liquidity = (self.market.ask1_lots if qty > 0 else self.market.bid1_lots)[t]
        self.fills.append(
            Fill(tick=t, qty=qty, price=price, liquidity=float(liquidity),
                 center_move=abs(price - self.center) / self.center * 1e4,
                 kind=kind)
        )
        self.pos += qty
        self.cash += cash_delta
        self.center = price
        self.last_fill_tick = t
        if qty > 0:
            self.cum_buys += 1
        else:
            self.cum_sells += 1
        self.last_px = price
        side = 1.0 if qty > 0 else -1.0
        self.int_notional += side * abs(qty) * (price - self.dec_mid)
        self.int_lots += abs(qty)

    def _immediate_fill(self, t: int, sell: float, buy: float, size: int) -> None:
        """新带宽已把现价甩在带外时，在决策点处按对手价立即成交一笔（3.5）。"""
        if not self.market.valid_book[t]:
            return
        bid, ask = self.market.bid1[t], self.market.ask1[t]
        if sell_crossed(bid, sell):
            lots = self._sell_lots(size)
            if lots > 0.0:
                self._execute(t, -lots, bid, "immediate")
        elif buy_crossed(ask, buy):
            lots = self._buy_lots(size, ask)
            if lots > 0.0:
                self._execute(t, lots, ask, "immediate")

    def _scan(self, lo: int, hi: int, sell: float | None, buy: float | None) -> tuple[int, int]:
        """在 (lo, hi] 内扫描首个触发，返回（触发 tick 或 hi, 方向 +1 买 / −1 卖 / 0 无）。

        任一侧为 None 表示该侧关闭（两侧均为 None 时区间必然无成交）。
        """
        m = self.market
        window = slice(lo + 1, hi + 1)
        hit = np.zeros(hi - lo, dtype=bool)
        if sell is not None:
            hit |= sell_crossed(m.bid1[window], sell)
        if buy is not None:
            hit |= buy_crossed(m.ask1[window], buy)
        hit &= m.scannable[window]
        if not hit.any():
            return hi, 0
        u = lo + 1 + int(np.argmax(hit))
        return u, -1 if sell is not None and sell_crossed(m.bid1[u], sell) else 1

    def _flatten_to_base(self, t: int, kind: str) -> None:
        """按对手方一档价把超额敞口一笔平回底仓（3.5：h=0 平仓与日终清仓共用此路径）。

        不模拟逐档穿价撮合，成交量不受一档深度限制。"""
        excess = self.pos - self.cfg.base_position
        if excess == 0.0:
            return
        price = (self.market.bid1 if excess > 0 else self.market.ask1)[t]
        self._execute(t, -excess, price, kind)

    # ---- 定长区间步进 ----
    def step(self, params: GridParams) -> StepResult:
        """按规则层参数推进一个定长决策区间（下一锚点或日终）。

        取参数而非动作档位：本环境是 design 第 3 节的规则层，与 RL 的档位表无关，
        固定参数基线因而可以用梯子之外的半宽（档位翻译见 action_params）。
        """
        cfg, m = self.cfg, self.market
        t = self.t
        width, size = params.width, params.size
        q0 = cfg.base_position

        minute_next = min(self.minute + cfg.decision_interval_min, MINUTES_PER_DAY - 1)
        t_next = m.decision_tick(minute_next)
        if minute_next >= MINUTES_PER_DAY - 1 or t_next <= t:
            t_next = m.n - 1
        done = t_next >= m.n - 1
        if done:
            t_next = m.n - 1

        equity_open = self.cash + self.pos * m.mid[t]
        fills_before = len(self.fills)

        # 新决策区间：p_dec 与区间成交统计重置（features §5.3 的成交过程通道）
        self.dec_mid = float(m.mid[t])
        self.last_px = 0.0
        self.int_notional = self.int_lots = 0.0

        sell = buy = None
        if width == 0.0:
            self._flatten_to_base(t, "flatten")
            size = 0                       # 平仓档不建网格，生效数量记 0
        else:
            sell, buy = self._lines(width)
            self._immediate_fill(t, sell, buy, size)
            sell, buy = self._lines(width)   # 立即触发后中心已移动，重算触发线

        # 逐笔扫至区间末：仓位 / 现金约束与触发线在每笔成交后重判
        excess_time = abs_time = boundary_time = 0.0
        max_abs = abs(self.pos - q0)
        scan_from = t
        while True:
            sell_lots = self._sell_lots(size) if width > 0.0 and size > 0 else 0.0
            buy_lots = self._buy_lots(size, buy) if width > 0.0 and size > 0 else 0.0
            u, side = self._scan(scan_from, t_next,
                                 sell if sell_lots > 0.0 else None,
                                 buy if buy_lots > 0.0 else None)
            self._write_priv(scan_from + 1, u, width, size)
            excess = self.pos - q0
            span = u - scan_from
            excess_time += excess * span
            abs_time += abs(excess) * span
            boundary_time += span if abs(excess) >= q0 else 0.0
            if side == 0:
                break
            if side < 0:
                self._execute(u, -sell_lots, sell, "grid")
            else:
                self._execute(u, buy_lots, buy, "grid")
            sell, buy = self._lines(width)
            self._write_priv(u, u, width, size)
            max_abs = max(max_abs, abs(self.pos - q0))
            scan_from = u

        if len(self.fills) == fills_before:
            self.center = m.mid[t_next]   # 区间无成交：决策锚点处中心改用当拍中间价（3.5）
        if done:
            self._flatten_to_base(t_next, "liquidate")
        self._write_priv(t_next, t_next, width, size)

        tau = max(t_next - t, 1)   # 起点已在日终的退化区间（盘中临停日）按 1 tick 计
        mean_excess = excess_time / tau
        p_open, p_close = m.mid[t], m.mid[t_next]
        equity_close = self.cash + self.pos * p_close
        b = self.base_value
        reward = ((equity_close - equity_open) - q0 * (p_close - p_open)) / b
        # 两个只用于训练的塑形项（6.2）：hindsight bonus 与存货惩罚的仓位取区间内
        # 时间加权。训练奖励整体按当日 σ_d 归一，使各交易日尺度一致。
        train_reward = reward
        if self.hindsight:
            train_reward += (cfg.hindsight_weight
                             * (m.hindsight_price(t) - p_open) * mean_excess / b)
        load = mean_excess * p_open / b
        train_reward = train_reward / m.sigma_d - cfg.inventory_lambda * load * load * tau / m.n

        self.intervals.append(
            Interval(tau=tau, excess=mean_excess, abs_excess=abs_time / tau,
                     max_abs_excess=max_abs, boundary_share=boundary_time / tau,
                     width=width, size=size, inventory_load=load * load)
        )
        self.n_steps += 1
        self.t = t_next
        self.minute = minute_next
        return StepResult(
            obs=None if done else m.observe(t_next, self.priv_window(t_next)),
            reward=float(reward),
            train_reward=float(train_reward),
            done=done,
            t=t_next,
            priv_hist=self.priv_window(t_next),
        )

    def net_value(self) -> float:
        """扣除底仓被动收益后的超额净值（以 1 为基准，见 6.1）。

        与逐区间奖励之和恒等：Σ r_t = [(期末权益 − E0) − Q0·(p_T − p_0)] / B。
        """
        m, q0 = self.market, self.cfg.base_position
        p_end = m.mid[m.n - 1]
        equity = self.cash + self.pos * p_end
        equity0 = self.cfg.max_position * self.p0
        return 1.0 + (equity - equity0 - q0 * (p_end - self.p0)) / self.base_value

    def _width_rel(self, tau: np.ndarray, chosen_width: np.ndarray) -> float:
        """当日时间加权生效半宽 / 当日开盘价：只计网格可触发的区间（平仓 0 与关闭档除外）。

        无可触发时间的日记 NaN，由汇总侧按有值日取均值。
        """
        cfg, m = self.cfg, self.market
        active = (chosen_width > 0.0) & (chosen_width < cfg.widths[-1])
        if not active.any():
            return float("nan")
        hw = np.array([half_width(w, m.atr, m.pre_close, cfg.window.min_width_ratio)
                       for w in chosen_width[active]])
        return float(tau[active] @ hw / (tau[active].sum() * m.open_px))

    def episode_log(self) -> dict:
        """当日补充指标（design 7.4）与选参所需的敞口负载（7.1）。

        成交口径为日内主动成交（网格触发、决策点立即成交与决策点平仓）；日终清仓
        是被动收尾，只报告其手数。每笔成交都重置中心，中心移动对全部日内成交有定义。
        """
        cfg = self.cfg
        intraday = [f for f in self.fills if f.kind != "liquidate"]
        flatten = [f for f in intraday if f.kind == "flatten"]
        qty = np.asarray([f.qty for f in intraday])
        liquidity = np.asarray([f.liquidity for f in intraday])
        center_move = np.asarray([f.center_move for f in intraday])
        tau = np.asarray([iv.tau for iv in self.intervals], dtype=np.float64)
        abs_excess = np.asarray([iv.abs_excess for iv in self.intervals])
        chosen_width = np.asarray([iv.width for iv in self.intervals])
        inventory_load = np.asarray([iv.inventory_load for iv in self.intervals])
        boundary_share = np.asarray([iv.boundary_share for iv in self.intervals])
        weight = tau / tau.sum()
        n_buy, n_sell = int((qty > 0).sum()), int((qty < 0).sum())
        turnover = float(np.abs(qty) @ np.asarray([f.price for f in intraday])) if len(qty) else 0.0
        equity0 = self.cfg.max_position * self.p0
        return {
            "n_fills": len(intraday),
            "n_immediate": sum(f.kind == "immediate" for f in intraday),
            "n_buys": n_buy,
            "n_sells": n_sell,
            "closure_rate": closure_rate(n_buy, n_sell),
            "n_decisions": len(self.intervals),
            "mean_tau": float(tau.mean()),
            "median_tau": float(np.median(tau)),
            "mean_fill_lots": float(np.abs(qty).mean()) if len(qty) else 0.0,
            "mean_center_move_bp": float(center_move.mean()) if len(center_move) else 0.0,
            "liquidity_ratio": (
                float(np.median(np.abs(qty) / np.maximum(liquidity, 1e-9)))
                if len(qty) else 0.0
            ),
            "max_abs_excess": float(max(iv.max_abs_excess for iv in self.intervals)),
            "time_weighted_excess": float(weight @ abs_excess),
            "inventory_load": float(weight @ inventory_load),
            "boundary_time": float(weight @ boundary_share),
            "width_rel": self._width_rel(tau, chosen_width),
            "width_time": [float(weight @ (chosen_width == h)) for h in cfg.widths],
            "n_flatten": len(flatten),
            "flattened_lots": float(sum(abs(f.qty) for f in flatten)),
            "turnover": turnover / equity0,
            "fee_cost": self.fee_cost / self.base_value,     # 相对 B 的当日显性成本
            "execution_cost": self.execution_cost / self.base_value,
            "liquidated_lots": float(sum(abs(f.qty) for f in self.fills if f.kind == "liquidate")),
        }
