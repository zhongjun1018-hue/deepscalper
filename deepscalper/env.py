"""交易环境：日内 MDP、限价单撮合模拟、含 hindsight 奖励与波动率标签。

论文 3.2 节 MDP 的 tick 适配：
  - 决策点为每个交易日的 lookback-1, lookback-1+20, ...（窗口不跨天）；
  - 动作为二元组 (穿价档深, 带符号委托手数)，数量为 0 表示不交易；
  - 撮合为逐档扫单：买单依次吃卖 1..k 档、卖单依次吃买 1..k 档，各档成交量受该档
    挂单量限制，深度不足时部分成交；日终以最深穿价档强平，残余仓位按中间价估值；
  - 奖励 r_t = leverage × [(p_{t+1}-p_t)·pos_t − cost_t] / cash_0，
    训练奖励附加 hindsight bonus：w·(p_{t+h}−p_t)·pos_t / cash_0（仅训练用）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .data import DayData
from .features import (
    PRIVATE_DIM,
    FeatureStats,
    build_macro_features,
    build_micro_matrix,
    future_price_index,
    volatility_label,
)


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
        self.bid_p = f[[f"Buy{i}Price" for i in range(1, 11)]].to_numpy(np.float64)
        self.ask_p = f[[f"Sell{i}Price" for i in range(1, 11)]].to_numpy(np.float64)
        bid_q = f[[f"Buy{i}OrderQty" for i in range(1, 11)]].to_numpy(np.float64)
        ask_q = f[[f"Sell{i}OrderQty" for i in range(1, 11)]].to_numpy(np.float64)
        # 挂单量由股折算为手；缺档（价格为 0）的深度记 0，使其不参与撮合
        self.bid_q = np.where(self.bid_p > 0, bid_q, 0.0) / cfg.lot_size
        self.ask_q = np.where(self.ask_p > 0, ask_q, 0.0) / cfg.lot_size
        self.mid = (self.bid_p[:, 0] + self.ask_p[:, 0]) / 2.0
        tvt = f["TotalVolumeTrade"].to_numpy(np.float64)
        self.vol_delta = np.diff(tvt, prepend=tvt[0])
        self.micro = build_micro_matrix(f)
        self.n = len(self.mid)
        lb = cfg.lookback_ticks
        self.decision_points = list(range(lb - 1, self.n, cfg.step_ticks))
        self.stats: FeatureStats | None = None

    @property
    def cash0(self) -> float:
        """账户初始现金：恰好可按首个决策点中间价买入最大持仓。"""
        return self.cfg.max_position * self.mid[self.decision_points[0]]

    def set_stats(self, stats: FeatureStats | None) -> None:
        """挂载标准化统计量并就地变换微观矩阵（仅调用一次）。"""
        if stats is not None:
            self.micro = stats.micro(self.micro)
        self.stats = stats

    @property
    def tradable(self) -> bool:
        return len(self.decision_points) > 1

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

        priv_hist 为 (micro_steps, 2) 的 (持仓, 现金) 历史，其行与微观序列的抽样
        时刻一一对应（决策间隔与抽样间隔同为 step_ticks），末行即决策点 t 的当期状态。
        """
        cfg = self.cfg
        times = np.arange(t - cfg.lookback_ticks + cfg.micro_stride, t + 1, cfg.micro_stride)
        private = np.empty((cfg.micro_steps, PRIVATE_DIM), dtype=np.float32)
        private[:, 0] = priv_hist[:, 0] / cfg.max_position
        private[:, 1] = priv_hist[:, 1] / self.cash0
        private[:, 2] = (self.n - 1 - times) / self.n
        return Observation(
            micro_lob=self.micro_window(t),
            private=private,
            macro=self.macro_at(t),
        )

    def execute(self, t: int, level: int, qty: float) -> tuple[float, float, float]:
        """在快照 t 处按穿价档深 level 扫单，返回（成交手数, 现金变动, 交易成本）。

        qty 为带符号委托手数（正买负卖）：买单依次吃卖 1..level 档、卖单依次吃买
        1..level 档，各档成交量受该档挂单量限制，深度不足时为部分成交。成本 =
        手续费 + 成交金额相对中间价估值的偏离（冲击成本），与现金账户记账完全一致。
        """
        if qty > 0:
            prices, depths = self.ask_p[t, :level], self.ask_q[t, :level]
        elif qty < 0:
            prices, depths = self.bid_p[t, :level], self.bid_q[t, :level]
        else:
            return 0.0, 0.0, 0.0
        # 逐档分配：第 k 档只承接前 k-1 档吃完后剩余的委托量
        take = np.clip(abs(qty) - np.concatenate(([0.0], np.cumsum(depths)[:-1])), 0.0, depths)
        filled, notional = float(take.sum()), float(take @ prices)
        if filled <= 0.0:
            return 0.0, 0.0, 0.0
        fee = notional * self.cfg.fee_rate
        side = 1.0 if qty > 0 else -1.0
        return side * filled, -side * notional - fee, fee + side * (notional - filled * self.mid[t])

    def vol_label(self, t: int) -> float:
        return volatility_label(self.mid, t, self.cfg.horizon_ticks, self.cfg.step_ticks)

    def hindsight_price(self, t: int) -> float:
        return self.mid[future_price_index(t, self.cfg.hindsight_ticks, self.n)]


@dataclass
class StepResult:
    obs: Observation | None
    reward: float            # 原始 P&L 奖励（评估口径）
    train_reward: float      # 含 hindsight bonus 的训练奖励
    done: bool
    t: int                   # 下一决策点（决策序列内索引）
    priv_hist: np.ndarray    # 下一决策点的 (持仓, 现金) 历史
    vol_label: float         # 当前决策点的波动率标签


class TradingEnv:
    """单交易日交易环境；每日结束强制平仓，次日重新开仓。"""

    def __init__(self, market: DayMarket, cfg: Config, hindsight: bool = True):
        self.market = market
        self.cfg = cfg
        self.hindsight = hindsight
        self.cash0 = market.cash0
        self.pos = 0.0
        self.cash = self.cash0
        self.step_idx = 0
        self.n_fills = 0
        # 首个决策点之前账户尚未交易，以空仓满现金作为私有历史的前置填充
        self.priv_hist = np.empty((cfg.micro_steps, 2), dtype=np.float32)
        self.priv_hist[:, 0] = 0.0
        self.priv_hist[:, 1] = self.cash0

    @property
    def t(self) -> int:
        return self.market.decision_points[self.step_idx]

    def observation(self) -> Observation:
        """当前决策点的观测。"""
        return self.market.observe(self.t, self.priv_hist)

    def _execute(self, t: int, level: int, qty: float) -> float:
        """按穿价档深撮合并更新账户，返回交易成本；委托量先按最大持仓约束截断。"""
        cfg = self.cfg
        if qty > 0:
            qty = min(qty, cfg.max_position - self.pos)
        elif qty < 0:
            qty = -min(-qty, self.pos + cfg.max_position)
        filled, cash_delta, cost = self.market.execute(t, level, qty)
        if filled != 0.0:
            self.pos += filled
            self.cash += cash_delta
            self.n_fills += 1
        return cost

    def step(self, action: tuple[int, int]) -> StepResult:
        cfg = self.cfg
        m = self.market
        t_now = self.t
        cost = self._execute(t_now, cfg.price_levels[action[0]], float(cfg.quantities[action[1]]))
        pos_held = self.pos  # 本决策区间实际持有的仓位（强平前）

        last = self.step_idx == len(m.decision_points) - 1
        t_next = m.n - 1 if last else m.decision_points[self.step_idx + 1]

        # 期末强平：以最深穿价档扫单，深度不足的残余仓位按中间价估值（见 net_value）
        if last and self.pos != 0.0:
            cost += self._execute(t_next, cfg.price_levels[-1], -self.pos)

        scale = cfg.leverage / self.cash0
        # 论文公式的真实账户口径：r_t = (p_{t+1} − p_t)·pos_t − 交易成本，
        # 成本含手续费与冲击成本；pos_t 为动作后的持仓
        reward = scale * ((m.mid[t_next] - m.mid[t_now]) * pos_held - cost)
        bonus = 0.0
        if self.hindsight:
            bonus = cfg.hindsight_weight * (m.hindsight_price(t_now) - m.mid[t_now]) * pos_held
        train_reward = reward + scale * bonus

        self.step_idx += 1
        self.priv_hist[:-1] = self.priv_hist[1:]
        self.priv_hist[-1] = (self.pos, self.cash)
        obs = None if last else m.observe(self.t, self.priv_hist)
        return StepResult(
            obs=obs,
            reward=float(reward),
            train_reward=float(train_reward),
            done=last,
            t=self.step_idx,
            priv_hist=self.priv_hist.copy(),
            vol_label=m.vol_label(t_now),
        )

    def net_value(self) -> float:
        """杠杆后的净值（以 1 为基准）；强平未成交的残余仓位按日末中间价估值。

        权益损失以初始现金为下限（破产吸收态），避免做空极端行情下净值为负。
        """
        raw = (self.cash + self.market.mid[self.market.n - 1] * self.pos - self.cash0) / self.cash0
        return max(0.0, 1.0 + self.cfg.leverage * raw)
