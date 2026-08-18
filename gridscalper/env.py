"""交易环境：成交驱动网格的 SMDP rollout、撮合模拟、奖励与日内统计。

design.md 第 3–7 节的实现：
  - 动作 (h, tilt, q) 给出两条触发线，向前扫描至触发或超时，区间时长 τ 由市场决定；
  - 决策点为「成交 / 超时 K tick / 日终」三者先到（4.1），中心只在成交时重置（3.3）；
  - 被动单以对手方一档严格穿越边界为成交条件，成交价为取整后的边界价（3.2）；
  - 奖励扣除底仓的被动收益，训练奖励另加 hindsight bonus 与存货惩罚（6.1–6.2）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config
from .data import DayData
from .features import (
    PRIV_CASH,
    PRIV_CENTER,
    PRIV_LAST_FILL,
    PRIV_POS,
    PRIV_RAW_DIM,
    PRIV_SIZE,
    PRIV_TILT,
    PRIV_WIDTH,
    PRIVATE_DIM,
    FeatureStats,
    build_macro_features,
    build_micro_matrix,
    future_price_index,
    volatility_label,
)
from .metrics import closure_rate

RANGE_TO_SIGMA = math.sqrt(8.0 / math.pi)  # 布朗运动 E[极差] = √(8/π)·σ ≈ 1.6σ
_TICK_EPS = 1e-9                           # 价位取整的浮点容差


@dataclass
class Observation:
    micro_lob: np.ndarray   # (micro_steps, MICRO_DIM)
    private: np.ndarray     # (micro_steps, PRIVATE_DIM)
    macro: np.ndarray       # (MACRO_DIM,)


class DayMarket:
    """单个交易日的预计算市场数据与观测构建器。"""

    def __init__(self, day: DayData, cfg: Config):
        f = day.frame
        self.cfg = cfg
        self.date = day.date
        self.atr = day.atr
        self.pre_close = day.pre_close
        self.bid_p = f[[f"Buy{i}Price" for i in range(1, 11)]].to_numpy(np.float64)
        self.ask_p = f[[f"Sell{i}Price" for i in range(1, 11)]].to_numpy(np.float64)
        bid_q = f[[f"Buy{i}OrderQty" for i in range(1, 11)]].to_numpy(np.float64)
        ask_q = f[[f"Sell{i}OrderQty" for i in range(1, 11)]].to_numpy(np.float64)
        # 挂单量由股折算为手；缺档（价格为 0）的深度记 0，使其不参与撮合
        self.bid_q = np.where(self.bid_p > 0, bid_q, 0.0) / cfg.lot_size
        self.ask_q = np.where(self.ask_p > 0, ask_q, 0.0) / cfg.lot_size
        self.mid = (self.bid_p[:, 0] + self.ask_p[:, 0]) / 2.0
        tvt = f["TotalVolumeTrade"].to_numpy(np.float64)
        self.vol_delta = np.maximum(np.diff(tvt, prepend=tvt[0]), 0.0)
        self.micro = build_micro_matrix(f)
        self.n = len(self.mid)
        self.start = cfg.lookback_ticks - 1

        # 参与扫描的快照：盘口未交叉，且累计成交笔数刷新历史高点。
        # fmax 累积使计数缺失或回退的快照不判成交，也不改变比较基准。
        num_trades = f["NumTrades"].to_numpy(np.float64)
        self.valid_book = self.bid_p[:, 0] <= self.ask_p[:, 0]
        self.scannable = np.empty(self.n, dtype=bool)
        self.scannable[0] = False
        self.scannable[1:] = (
            self.valid_book[1:]
            & (num_trades[1:] > np.fmax.accumulate(num_trades)[:-1])
        )
        # 拟合标准化统计量所用的固定 tick 网格采样点（与决策点无关）
        self.sample_points = list(range(self.start, self.n, cfg.micro_stride))
        self.stats: FeatureStats | None = None

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
        private[:, 4] = priv_hist[:, PRIV_WIDTH] / cfg.half_widths[-1]
        private[:, 5] = priv_hist[:, PRIV_TILT] / cfg.max_tilt
        private[:, 6] = priv_hist[:, PRIV_SIZE] / cfg.sizes[-1]
        private[:, 7] = (times - priv_hist[:, PRIV_LAST_FILL]) / self.n
        return Observation(
            micro_lob=self.micro_window(t),
            private=private,
            macro=self.macro_at(t),
        )

    def trade(self, t: int, qty: float, price: float) -> tuple[float, float, float]:
        """在快照 t 处按给定价格成交 qty 手（正买负卖）。

        返回（现金变动, 手续费, 执行成本）：手续费为显性费用，执行成本为成交价相对
        中间价的偏离。两者之和即 3.2 的交易成本，与现金账户记账完全一致。
        """
        side = 1.0 if qty > 0 else -1.0
        notional = abs(qty) * price
        fee = notional * self.cfg.fee_rate(side)
        return -side * notional - fee, fee, side * (notional - abs(qty) * self.mid[t])

    def sweep(self, t: int, qty: float) -> tuple[float, float, float, float]:
        """日终平仓的逐档扫单：返回（成交手数, 现金变动, 手续费, 执行成本）。

        买单依次吃卖 1..10 档、卖单依次吃买 1..10 档，各档成交量受该档挂单量限制，
        深度不足时部分成交。
        """
        if qty > 0:
            prices, depths = self.ask_p[t], self.ask_q[t]
        else:
            prices, depths = self.bid_p[t], self.bid_q[t]
        # 逐档分配：第 k 档只承接前 k-1 档吃完后剩余的委托量
        take = np.clip(abs(qty) - np.concatenate(([0.0], np.cumsum(depths)[:-1])), 0.0, depths)
        filled, notional = float(take.sum()), float(take @ prices)
        if filled <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        side = 1.0 if qty > 0 else -1.0
        fee = notional * self.cfg.fee_rate(side)
        return side * filled, -side * notional - fee, fee, side * (notional - filled * self.mid[t])

    def vol_label(self, t: int) -> float:
        return volatility_label(self.mid, t, self.cfg.horizon_ticks, self.cfg.micro_stride)

    def hindsight_price(self, t: int) -> float:
        return self.mid[future_price_index(t, self.cfg.hindsight_ticks, self.n)]


@dataclass(frozen=True)
class GridParams:
    """规则层参数：一次决策所设定的网格形状（design 第 3 节）。"""

    width: float        # 整体半宽（× ATR3），不必落在动作梯子上
    tilt: int           # 倾斜档，取 ±max_tilt 时对侧关闭
    size: int           # 每次触发成交手数


def action_params(cfg: Config, action: tuple[int, int, int]) -> GridParams:
    """把三分支的动作档位索引翻译为规则层参数。"""
    return GridParams(cfg.half_widths[action[0]], cfg.tilts[action[1]], cfg.sizes[action[2]])


@dataclass
class Fill:
    tick: int
    qty: float          # 带符号成交手数（正买负卖）
    price: float
    liquidity: float    # 成交快照的对手一档深度（手）
    center_move: float  # 成交价相对上一中心的移动距离（bp）
    immediate: bool     # 是否为决策点处的立即触发（3.3）


@dataclass
class Interval:
    tau: int
    excess: float       # 区间内恒定的超额敞口 pos − Q0（手）
    width: float        # 生效半宽档（× ATR3）
    tilt: int
    size: int
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
    vol_label: float         # 当前决策点的波动率标签


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
        self.liquidated = 0.0
        # 私有状态按 tick 记录：决策点由事件触发，回看窗口的抽样时刻不再落在决策点上
        self.priv_raw = np.empty((market.n, PRIV_RAW_DIM), dtype=np.float32)
        self._write_priv(0, self.t, width=0.0, tilt=0, size=0)
        # 建网前的填充段不携带网格状态：中心偏离与距上次成交时长均记 0
        self.priv_raw[: self.t + 1, PRIV_CENTER] = market.mid[: self.t + 1]
        self.priv_raw[: self.t + 1, PRIV_LAST_FILL] = np.arange(self.t + 1)

    # ---- 私有状态 ----
    def _write_priv(self, lo: int, hi: int, width: float, tilt: int, size: int) -> None:
        """将当前账户与网格状态写入 tick 区间 [lo, hi]。"""
        rows = self.priv_raw[lo : hi + 1]
        rows[:, PRIV_POS] = self.pos
        rows[:, PRIV_CASH] = self.cash
        rows[:, PRIV_CENTER] = self.center
        rows[:, PRIV_WIDTH] = width
        rows[:, PRIV_TILT] = tilt
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
    def _lines(self, width: float, tilt: int) -> tuple[float | None, float | None]:
        """当前中心下的两条触发线；倾斜取端点时对侧关闭（返回 None）。

        卖出线向上取整、买入线向下取整到最小变动价位，保证是合法限价且不缩窄名义半宽。
        """
        cfg = self.cfg
        span = max(width * self.market.atr, cfg.min_half_width_ratio * self.center)
        sell = buy = None
        if tilt != -cfg.max_tilt:
            sell = _to_tick(self.center + span * cfg.tilt_ratio ** (-tilt), cfg.tick_size, up=True)
        if tilt != cfg.max_tilt:
            buy = _to_tick(self.center - span * cfg.tilt_ratio ** tilt, cfg.tick_size, up=False)
        return sell, buy

    def _sell_lots(self, size: int) -> float:
        """卖出侧的有效成交量：受仓位下界截断，余额不足最小申报量时一次性卖出。"""
        return min(float(size), self.pos)

    def _buy_lots(self, size: int, price: float) -> float:
        """买入侧的有效成交量：受仓位上界与现金约束，不足最小申报量时该侧失效。"""
        cfg = self.cfg
        lots = min(float(size), cfg.max_position - self.pos)
        if lots < cfg.min_order_lots:
            return 0.0
        if self.cash < lots * price * (1.0 + cfg.commission_rate):
            return 0.0
        return lots

    def _fill(self, t: int, qty: float, price: float, immediate: bool) -> float:
        """成交一笔并重置中心，返回交易成本。"""
        cash_delta, fee, execution = self.market.trade(t, qty, price)
        self.fee_cost += fee
        self.execution_cost += execution
        liquidity = (self.market.ask_q if qty > 0 else self.market.bid_q)[t, 0]
        self.fills.append(
            Fill(tick=t, qty=qty, price=price, liquidity=float(liquidity),
                 center_move=abs(price - self.center) / self.center * 1e4,
                 immediate=immediate)
        )
        self.pos += qty
        self.cash += cash_delta
        self.center = price
        self.last_fill_tick = t
        return fee + execution

    def _immediate_fill(self, t: int, sell: float | None, buy: float | None, size: int) -> float:
        """新带宽已把现价甩在带外时，在决策点处按对手价立即成交一笔（3.3）。"""
        if not self.market.valid_book[t]:
            return 0.0
        bid, ask = self.market.bid_p[t, 0], self.market.ask_p[t, 0]
        if sell is not None and bid > sell:
            lots = self._sell_lots(size)
            if lots > 0.0:
                return self._fill(t, -lots, bid, immediate=True)
        elif buy is not None and ask < buy:
            lots = self._buy_lots(size, ask)
            if lots > 0.0:
                return self._fill(t, lots, ask, immediate=True)
        return 0.0

    def _scan(self, t: int, sell: float | None, buy: float | None) -> tuple[int, int]:
        """自 t+1 起扫描至触发或超时，返回（区间终点 tick, 触发方向 +1 买 / −1 卖 / 0 超时）。"""
        m = self.market
        end = min(t + self.cfg.timeout_ticks, m.n - 1)
        window = slice(t + 1, end + 1)
        hit = np.zeros(end - t, dtype=bool)
        if sell is not None:
            hit |= m.bid_p[window, 0] > sell
        if buy is not None:
            hit |= m.ask_p[window, 0] < buy
        hit &= m.scannable[window]
        if not hit.any():
            return end, 0
        u = t + 1 + int(np.argmax(hit))
        return u, -1 if sell is not None and m.bid_p[u, 0] > sell else 1

    def _liquidate(self, t: int) -> float:
        """日终平回底仓；申报量或盘口深度不足的残余按中间价估值（3.5）。"""
        excess = self.pos - self.cfg.base_position
        if excess == 0.0:
            return 0.0
        if abs(excess) < self.cfg.min_order_lots:
            return 0.0
        filled, cash_delta, fee, execution = self.market.sweep(t, -excess)
        self.pos += filled
        self.cash += cash_delta
        self.fee_cost += fee
        self.execution_cost += execution
        self.liquidated = abs(filled)
        return fee + execution

    # ---- SMDP 步进 ----
    def step(self, params: GridParams) -> StepResult:
        """按规则层参数推进一个决策区间。

        取参数而非动作档位：本环境是 design 第 3 节的规则层，与 RL 的档位表无关，
        固定参数基线因而可以用梯子之外的半宽（档位翻译见 action_params）。
        """
        cfg, m = self.cfg, self.market
        t = self.t
        width, tilt, size = params.width, params.tilt, params.size
        if size == 0:
            width, tilt = 0.0, 0

        sell, buy = self._lines(width, tilt)
        cost = 0.0
        if size > 0:
            cost += self._immediate_fill(t, sell, buy, size)
            sell, buy = self._lines(width, tilt)   # 立即触发后中心已移动，重算触发线
        # 区间内 pos 恒定，故两侧的有效成交量与开关只需判定一次
        sell_lots = self._sell_lots(size)
        buy_lots = self._buy_lots(size, buy) if buy is not None else 0.0
        pos_held = self.pos

        t_next, side = self._scan(
            t, sell if sell_lots > 0.0 else None, buy if buy_lots > 0.0 else None
        )
        tau = t_next - t
        self._write_priv(t + 1, t_next, width, tilt, size)  # 区间内恒定的私有状态

        if side < 0:
            cost += self._fill(t_next, -sell_lots, sell, immediate=False)
        elif side > 0:
            cost += self._fill(t_next, buy_lots, buy, immediate=False)
        done = t_next >= m.n - 1
        if done:
            cost += self._liquidate(t_next)
        self._write_priv(t_next, t_next, width, tilt, size)  # 区间末成交后的状态

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
            Interval(tau=tau, excess=excess, width=width, tilt=tilt, size=size,
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
            vol_label=m.vol_label(t),
        )

    def net_value(self) -> float:
        """扣除底仓被动收益后的超额净值（以 1 为基准，见 6.1）。

        与逐区间奖励之和恒等：Σ r_t = [(期末权益 − E0) − Q0·(p_T − p_0)] / B。
        """
        m, q0 = self.market, self.cfg.base_position
        p_end = m.mid[m.n - 1]
        equity = self.cash + self.pos * p_end
        return 1.0 + (equity - m.equity0 - q0 * (p_end - m.p0)) / m.base_value

    def episode_log(self) -> dict:
        """当日补充指标（design 7.4）与选参所需的敞口负载（7.1）。"""
        cfg, m = self.cfg, self.market
        q0 = cfg.base_position
        qty = np.asarray([f.qty for f in self.fills])
        liquidity = np.asarray([f.liquidity for f in self.fills])
        center_move = np.asarray([f.center_move for f in self.fills])
        tau = np.asarray([iv.tau for iv in self.intervals], dtype=np.float64)
        excess = np.asarray([iv.excess for iv in self.intervals])
        chosen_width = np.asarray([iv.width for iv in self.intervals])
        chosen_size = np.asarray([iv.size for iv in self.intervals])
        chosen_tilt = np.asarray([iv.tilt for iv in self.intervals])
        inventory_load = np.asarray([iv.inventory_load for iv in self.intervals])
        weight = tau / tau.sum()
        n_buy, n_sell = int((qty > 0).sum()), int((qty < 0).sum())
        turnover = float(np.abs(qty) @ np.asarray([f.price for f in self.fills])) if len(qty) else 0.0
        return {
            "n_fills": len(self.fills),
            "n_immediate": int(sum(f.immediate for f in self.fills)),
            "n_buys": n_buy,
            "n_sells": n_sell,
            "closure_rate": closure_rate(n_buy, n_sell),
            "n_decisions": len(self.intervals),
            "mean_tau": float(tau.mean()),
            "median_tau": float(np.median(tau)),
            "mean_fill_lots": float(np.abs(qty).mean()) if len(qty) else 0.0,
            "mean_center_move_bp": float(center_move.mean()) if len(qty) else 0.0,
            "liquidity_ratio": (
                float(np.median(np.abs(qty) / np.maximum(liquidity, 1e-9)))
                if len(qty) else 0.0
            ),
            "max_abs_excess": float(np.abs(excess).max()),
            "time_weighted_excess": float(weight @ np.abs(excess)),
            "inventory_load": float(weight @ inventory_load),
            "boundary_time": float(weight @ ((excess <= -q0) | (excess >= q0))),
            "width_time": [float(weight @ (chosen_width == h)) for h in cfg.half_widths],
            "tilt_time": [float(weight @ (chosen_tilt == v)) for v in cfg.tilts],
            "size_time": [float(weight @ (chosen_size == s)) for s in cfg.sizes],
            "turnover": turnover / m.equity0,
            "fee_cost": self.fee_cost / m.base_value,        # 相对 B 的当日显性成本
            "execution_cost": self.execution_cost / m.base_value,
            "liquidated_lots": self.liquidated,
        }


def _to_tick(price: float, tick: float, up: bool) -> float:
    """将价格取整到最小变动价位（up 为向上取整，否则向下）。"""
    steps = math.ceil(price / tick - _TICK_EPS) if up else math.floor(price / tick + _TICK_EPS)
    return steps * tick
