"""交易环境：成交驱动网格的 SMDP rollout、撮合模拟、奖励与日内统计。

design.md 第 3–7 节的实现：
  - 动作 (h, q) 给出对称的两条触发线与每笔成交手数，向前扫描至触发或超时，
    区间时长 τ 由市场决定；
  - h = 0 时在决策点按对手方一档价平回底仓（3.5，仅净持仓非零时可选），
    h 极大（关闭档）则网格无法触发，超时机制保证关闭档不会形成吸收态；
  - 决策点为「成交 / 超时 K tick / 日终」三者先到（4.1）；每笔成交后中心重置为
    成交价，超时未成交则重置为当拍中间价（3.5）；
  - 被动单以对手方一档严格穿越边界为成交条件，成交价为取整后的边界价（3.2）；
  - 奖励扣除底仓的被动收益，训练奖励另加 hindsight bonus 与存货惩罚（6.1–6.2）。

网格几何（半宽、边界取整、严格穿越判定）与 strategy/grid.py 共用同一定义，
显性费用一律走 strategy/costs.py。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from data_provider.ticks import DayData
from strategy.costs import COMMISSION_RATE, fee
from strategy.grid import boundaries, buy_crossed, half_width, sell_crossed
from strategy.metrics import closure_rate

from .config import Config
from .features import (
    PRIV_CASH,
    PRIV_CENTER,
    PRIV_LAST_FILL,
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
    future_price_index,
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

    window 为该日的样本行块（R×29：24 维窗口统计 + 5 维前瞻预测，行索引即 tick 索引，
    见 data/features.md §3 与 §5.2）。R 为该标的单日最大快照数，故对本日任一 tick 恒可
    索引；缺省为零块（无缓存时的降级形态，如冒烟测试的合成数据）。
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
        self.start = cfg.lookback_ticks - 1

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
        # 拟合标准化统计量所用的固定 tick 网格采样点（与决策点无关）
        self.sample_points = list(range(self.start, self.n, cfg.micro_stride))
        self.stats: FeatureStats | None = None
        self.symbol_id = symbol_id
        self.window = window if window is not None else np.zeros(
            (self.n, WINDOW_DIM + PRED_DIM), dtype=np.float32
        )

    @property
    def tradable(self) -> bool:
        """需有有效 ATR（前 A 日完整）与至少一个可推进的决策点。"""
        return bool(np.isfinite(self.atr)) and self.atr > 0.0 and self.n > self.cfg.lookback_ticks

    @property
    def p0(self) -> float:
        """建仓参考价：首个决策点的中间价。"""
        return float(self.mid[self.start])

    @property
    def equity0(self) -> float:
        """初始权益 E0 = 2Q0·p0（底仓 Q0 手 + 等额现金）。"""
        return self.cfg.max_position * self.p0

    @property
    def base_value(self) -> float:
        """归一基准 B = Q0·p0，即最大单边超额敞口所对应的资本（design 3.5）。"""
        return self.cfg.base_position * self.p0

    @property
    def sigma_d(self) -> float:
        """当日收益率标准差的估计 σ_d = ATR3 / (1.6·前收)（design 6.2）。"""
        return self.atr / (RANGE_TO_SIGMA * self.pre_close)

    def set_stats(self, stats: FeatureStats | None) -> None:
        """挂载标准化统计量并就地变换微观矩阵（仅调用一次）。"""
        if stats is not None:
            self.micro = stats.micro(self.micro)
        self.stats = stats

    def sample_times(self, t: int) -> np.ndarray:
        """决策点 t 的回看抽样时刻：末项恰为 t，与微观序列、私有序列逐行对应。"""
        cfg = self.cfg
        return np.arange(t - cfg.lookback_ticks + cfg.micro_stride, t + 1, cfg.micro_stride)

    def micro_window(self, t: int) -> np.ndarray:
        """回看窗口内按 micro_stride 抽样的微观序列 (micro_steps, MICRO_DIM)。

        取每个抽样区间的末帧，故序列末帧恰为决策点 t 的快照，与宏观 bar 的
        收盘价对齐——决策所依据的最新盘口与撮合所用的盘口为同一快照。
        """
        cfg = self.cfg
        lo = t - cfg.lookback_ticks + cfg.micro_stride
        return self.micro[lo : t + 1 : cfg.micro_stride]

    def macro_at(self, t: int, normalized: bool = True) -> np.ndarray:
        cfg = self.cfg
        lo = t - cfg.lookback_ticks + 1
        macro = build_macro_features(
            self.mid[lo : t + 1], self.vol_delta[lo : t + 1], cfg.n_bars
        )
        # 行索引即 tick 索引：第 t 行的窗口恰好收于 t，不含未完结数据
        macro = np.concatenate([macro, self.window[t]])
        if normalized and self.stats is not None:
            macro = self.stats.macro(macro)
        return macro

    def observe(self, t: int, priv_hist: np.ndarray) -> Observation:
        """构建决策点 t 的观测。

        priv_hist 为 (micro_steps, PRIV_RAW_DIM) 的私有状态原始记录，其行与 sample_times(t)
        一一对应（按固定 tick 网格抽样，与事件驱动的决策时刻无关），末行即 t 的当期状态。
        """
        cfg = self.cfg
        times = self.sample_times(t)
        private = np.empty((cfg.micro_steps, PRIVATE_DIM), dtype=np.float32)
        private[:, 0] = priv_hist[:, PRIV_POS] / cfg.max_position
        private[:, 1] = priv_hist[:, PRIV_CASH] / self.equity0
        private[:, 2] = (self.n - 1 - times) / self.n
        private[:, 3] = (self.mid[times] - priv_hist[:, PRIV_CENTER]) / self.atr
        private[:, 4] = priv_hist[:, PRIV_WIDTH] / cfg.widths[-1]
        private[:, 5] = priv_hist[:, PRIV_SIZE] / cfg.sizes[-1]
        private[:, 6] = (times - priv_hist[:, PRIV_LAST_FILL]) / self.n
        return Observation(
            micro_lob=self.micro_window(t),
            private=private,
            macro=self.macro_at(t),
            symbol_id=self.symbol_id,
            flatten_allowed=bool(priv_hist[-1, PRIV_POS] != cfg.base_position),
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
        return self.mid[future_price_index(t, self.cfg.hindsight_ticks, self.n)]


@dataclass(frozen=True)
class GridParams:
    """规则层参数：一次决策所设定的网格状态（design 第 3 节）。"""

    width: float        # 对称半宽（× ATR3），0 表示平回底仓，不必落在动作梯子上
    size: int           # 每次触发成交手数


def action_params(cfg: Config, action: tuple[int, int]) -> GridParams:
    """把两分支的动作档位索引（半宽档, 数量档）翻译为规则层参数。"""
    return GridParams(cfg.widths[action[0]], cfg.sizes[action[1]])


@dataclass
class Fill:
    tick: int
    qty: float          # 带符号成交手数（正买负卖）
    price: float
    liquidity: float    # 成交快照的对手一档深度（手）
    center_move: float  # 成交价相对上一中心的移动距离（bp）
    kind: str           # "grid" 区间末触发 / "immediate" 决策点立即成交 /
                        # "flatten" 决策点平仓 / "liquidate" 日终清仓（3.5）


@dataclass
class Interval:
    tau: int
    excess: float       # 区间内恒定的超额敞口 pos − Q0（手）
    width: float        # 生效半宽档（× ATR3）
    size: int           # 生效成交手数（平仓时记 0）
    inventory_load: float  # 区间起点的 (I/B)^2


@dataclass
class StepResult:
    obs: Observation | None
    reward: float            # 超额 P&L 奖励（评估口径）
    train_reward: float      # 按 σ_d 归一、含 hindsight bonus 与存货惩罚的训练奖励
    done: bool
    t: int                   # 下一决策点的 tick 索引
    tau: int                 # 本决策区间时长（tick）
    priv_hist: np.ndarray    # 下一决策点的私有状态历史


class TradingEnv:
    """单交易日的网格环境；开盘持底仓 Q0，日终只平回净敞口。"""

    def __init__(self, market: DayMarket, hindsight: bool = True):
        self.market = market
        self.cfg = market.cfg   # 与市场同源，避免两份配置静默分歧
        self.hindsight = hindsight
        self.t = market.start
        self.pos = float(self.cfg.base_position)
        self.cash = market.base_value      # 建立底仓后的剩余现金 Q0·p0
        self.center = market.mid[self.t]   # 首个决策点以中间价建网，该快照只建网不成交
        self.last_fill_tick = self.t
        self.n_steps = 0
        self.fee_cost = 0.0        # 当日累计手续费（显性）
        self.execution_cost = 0.0  # 当日累计成交价相对中间价的偏离
        self.fills: list[Fill] = []
        self.intervals: list[Interval] = []
        # 私有状态按 tick 记录：决策点由事件触发，回看窗口的抽样时刻不再落在决策点上
        self.priv_raw = np.empty((market.n, PRIV_RAW_DIM), dtype=np.float32)
        self._write_priv(0, self.t, width=0.0, size=0)
        # 建网前的填充段不携带网格状态：中心偏离与距上次成交时长均记 0
        self.priv_raw[: self.t + 1, PRIV_CENTER] = market.mid[: self.t + 1]
        self.priv_raw[: self.t + 1, PRIV_LAST_FILL] = np.arange(self.t + 1)

    # ---- 私有状态 ----
    def _write_priv(self, lo: int, hi: int, width: float, size: int) -> None:
        """将当前账户与网格状态写入 tick 区间 [lo, hi]。"""
        rows = self.priv_raw[lo : hi + 1]
        rows[:, PRIV_POS] = self.pos
        rows[:, PRIV_CASH] = self.cash
        rows[:, PRIV_CENTER] = self.center
        rows[:, PRIV_WIDTH] = width
        rows[:, PRIV_SIZE] = size
        rows[:, PRIV_LAST_FILL] = self.last_fill_tick

    def priv_window(self, t: int) -> np.ndarray:
        """决策点 t 的私有状态历史 (micro_steps, PRIV_RAW_DIM)。"""
        cfg = self.cfg
        lo = t - cfg.lookback_ticks + cfg.micro_stride
        return self.priv_raw[lo : t + 1 : cfg.micro_stride].copy()

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

    def _execute(self, t: int, qty: float, price: float, kind: str) -> float:
        """按给定价格成交 qty 手、登记成交并记账，返回交易成本。

        成交后中心重置到成交价（3.5），下一决策以该价为网格中心。
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
        return fee_cost + execution

    def _immediate_fill(self, t: int, sell: float, buy: float, size: int) -> float:
        """新带宽已把现价甩在带外时，在决策点处按对手价立即成交一笔（3.5）。"""
        if not self.market.valid_book[t]:
            return 0.0
        bid, ask = self.market.bid1[t], self.market.ask1[t]
        if sell_crossed(bid, sell):
            lots = self._sell_lots(size)
            if lots > 0.0:
                return self._execute(t, -lots, bid, "immediate")
        elif buy_crossed(ask, buy):
            lots = self._buy_lots(size, ask)
            if lots > 0.0:
                return self._execute(t, lots, ask, "immediate")
        return 0.0

    def _scan(self, t: int, sell: float | None, buy: float | None) -> tuple[int, int]:
        """自 t+1 起扫描至触发或超时，返回（区间终点 tick, 触发方向 +1 买 / −1 卖 / 0 超时）。

        任一侧为 None 表示该侧关闭（网格关闭时两侧均为 None，区间必然超时结束）。
        """
        m = self.market
        end = min(t + self.cfg.timeout_ticks, m.n - 1)
        window = slice(t + 1, end + 1)
        hit = np.zeros(end - t, dtype=bool)
        if sell is not None:
            hit |= sell_crossed(m.bid1[window], sell)
        if buy is not None:
            hit |= buy_crossed(m.ask1[window], buy)
        hit &= m.scannable[window]
        if not hit.any():
            return end, 0
        u = t + 1 + int(np.argmax(hit))
        return u, -1 if sell is not None and sell_crossed(m.bid1[u], sell) else 1

    def _flatten_to_base(self, t: int, kind: str) -> float:
        """按对手方一档价把超额敞口一笔平回底仓，返回交易成本（3.5：h=0 平仓与日终清仓
        共用此路径）。不模拟逐档穿价撮合，成交量不受一档深度限制。"""
        excess = self.pos - self.cfg.base_position
        if excess == 0.0:
            return 0.0
        price = (self.market.bid1 if excess > 0 else self.market.ask1)[t]
        return self._execute(t, -excess, price, kind)

    # ---- SMDP 步进 ----
    def step(self, params: GridParams) -> StepResult:
        """按规则层参数推进一个决策区间。

        取参数而非动作档位：本环境是 design 第 3 节的规则层，与 RL 的档位表无关，
        固定参数基线因而可以用梯子之外的半宽（档位翻译见 action_params）。
        """
        cfg, m = self.cfg, self.market
        t = self.t
        width, size = params.width, params.size

        sell = buy = None
        cost = 0.0
        if width == 0.0:
            cost += self._flatten_to_base(t, "flatten")
            size = 0                       # 平仓档不建网格，生效数量记 0
        else:
            sell, buy = self._lines(width)
            cost += self._immediate_fill(t, sell, buy, size)
            sell, buy = self._lines(width)   # 立即触发后中心已移动，重算触发线
        # 区间内 pos 恒定，故两侧的有效成交量只需判定一次
        sell_lots = self._sell_lots(size) if size > 0 else 0.0
        buy_lots = self._buy_lots(size, buy) if size > 0 else 0.0
        pos_held = self.pos

        t_next, side = self._scan(
            t, sell if sell_lots > 0.0 else None, buy if buy_lots > 0.0 else None
        )
        tau = t_next - t
        self._write_priv(t + 1, t_next, width, size)  # 区间内恒定的私有状态

        if side < 0:
            cost += self._execute(t_next, -sell_lots, sell, "grid")
        elif side > 0:
            cost += self._execute(t_next, buy_lots, buy, "grid")
        else:
            self.center = m.mid[t_next]   # 连续 K tick 未成交：中心改用当拍中间价（3.5）
        done = t_next >= m.n - 1
        if done:
            cost += self._flatten_to_base(t_next, "liquidate")
        self._write_priv(t_next, t_next, width, size)  # 区间末成交后的状态

        excess = pos_held - cfg.base_position
        b = m.base_value
        reward = ((m.mid[t_next] - m.mid[t]) * excess - cost) / b
        # 两个只用于训练的塑形项：hindsight bonus 按 τ/K 加权，存货惩罚按区间方差计（6.2）。
        # 训练奖励整体按当日 σ_d 归一，使各交易日尺度一致；归一后区间方差即 load²·τ/N。
        train_reward = reward
        if self.hindsight:
            train_reward += (cfg.hindsight_weight * tau / cfg.timeout_ticks
                             * (m.hindsight_price(t) - m.mid[t]) * excess / b)
        load = excess * m.mid[t] / b
        train_reward = train_reward / m.sigma_d - cfg.inventory_lambda * load * load * tau / m.n

        self.intervals.append(
            Interval(tau=tau, excess=excess, width=width, size=size,
                     inventory_load=load * load)
        )
        self.n_steps += 1
        self.t = t_next
        return StepResult(
            obs=None if done else m.observe(t_next, self.priv_window(t_next)),
            reward=float(reward),
            train_reward=float(train_reward),
            done=done,
            t=t_next,
            tau=tau,
            priv_hist=self.priv_window(t_next),
        )

    def net_value(self) -> float:
        """扣除底仓被动收益后的超额净值（以 1 为基准，见 6.1）。

        与逐区间奖励之和恒等：Σ r_t = [(期末权益 − E0) − Q0·(p_T − p_0)] / B。
        """
        m, q0 = self.market, self.cfg.base_position
        p_end = m.mid[m.n - 1]
        equity = self.cash + self.pos * p_end
        return 1.0 + (equity - m.equity0 - q0 * (p_end - m.p0)) / m.base_value

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
        cfg, m = self.cfg, self.market
        q0 = cfg.base_position
        intraday = [f for f in self.fills if f.kind != "liquidate"]
        flatten = [f for f in intraday if f.kind == "flatten"]
        qty = np.asarray([f.qty for f in intraday])
        liquidity = np.asarray([f.liquidity for f in intraday])
        center_move = np.asarray([f.center_move for f in intraday])
        tau = np.asarray([iv.tau for iv in self.intervals], dtype=np.float64)
        excess = np.asarray([iv.excess for iv in self.intervals])
        chosen_width = np.asarray([iv.width for iv in self.intervals])
        inventory_load = np.asarray([iv.inventory_load for iv in self.intervals])
        weight = tau / tau.sum()
        n_buy, n_sell = int((qty > 0).sum()), int((qty < 0).sum())
        turnover = float(np.abs(qty) @ np.asarray([f.price for f in intraday])) if len(qty) else 0.0
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
            "max_abs_excess": float(np.abs(excess).max()),
            "time_weighted_excess": float(weight @ np.abs(excess)),
            "inventory_load": float(weight @ inventory_load),
            "boundary_time": float(weight @ ((excess <= -q0) | (excess >= q0))),
            "width_rel": self._width_rel(tau, chosen_width),
            "width_time": [float(weight @ (chosen_width == h)) for h in cfg.widths],
            "n_flatten": len(flatten),
            "flattened_lots": float(sum(abs(f.qty) for f in flatten)),
            "turnover": turnover / m.equity0,
            "fee_cost": self.fee_cost / m.base_value,        # 相对 B 的当日显性成本
            "execution_cost": self.execution_cost / m.base_value,
            "liquidated_lots": float(sum(abs(f.qty) for f in self.fills if f.kind == "liquidate")),
        }
