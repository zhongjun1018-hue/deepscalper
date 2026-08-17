"""交易环境：日内 MDP、限价单撮合模拟、含 hindsight 奖励与波动率标签。

论文 3.2 节 MDP 的 tick 适配：
  - 决策点为每个交易日的 lookback-1, lookback-1+20, ...（窗口不跨天）；
  - 动作为二元组 (价位偏移, 带符号目标数量)，数量为 0 表示不交易；价位偏移
    相对对手价（买参考卖一价、卖参考买一价）；
  - 撮合简化：买单限价 ≥ 卖一价则以卖一价成交，卖单限价 ≤ 买一价则以买一价成交，
    未穿价则不成交；收盘强制以最后中间价平仓；
  - 奖励 r_t = leverage × [(p_{t+1}-p_t)·pos_t − cost_t] / cash_0，
    训练奖励附加 hindsight bonus：w·(p_{t+h}−p_t)·pos_t / cash_0（仅训练用）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .data import DayData
from .features import (
    MACRO_DIM,
    MICRO_DIM,
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
        self.date = day.date
        self.cfg = cfg
        self.bid1 = f["Buy1Price"].to_numpy(np.float64)
        self.ask1 = f["Sell1Price"].to_numpy(np.float64)
        self.mid = (self.bid1 + self.ask1) / 2.0
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

    def macro_at(self, t: int, normalized: bool = True) -> np.ndarray:
        cfg = self.cfg
        lo = t - cfg.lookback_ticks + 1
        macro = build_macro_features(
            self.mid[lo : t + 1], self.vol_delta[lo : t + 1], cfg.n_bars
        )
        if normalized and self.stats is not None:
            macro = self.stats.macro(macro)
        return macro

    def observe(self, t: int, pos: float, cash: float, cash0: float) -> Observation:
        cfg = self.cfg
        lo = t - cfg.lookback_ticks + 1
        micro_lob = self.micro[lo : t + 1 : cfg.micro_stride]
        priv = np.array(
            [pos / cfg.max_position, cash / cash0, (self.n - 1 - t) / self.n],
            dtype=np.float32,
        )
        return Observation(
            micro_lob=micro_lob,
            private=np.repeat(priv[None, :], cfg.micro_steps, axis=0),
            macro=self.macro_at(t),
        )

    def vol_label(self, t: int) -> float:
        return volatility_label(self.mid, t, self.cfg.horizon_ticks, self.cfg.step_ticks)

    def hindsight_price(self, t: int) -> float:
        return self.mid[future_price_index(t, self.cfg.hindsight_ticks, self.n)]


@dataclass
class StepResult:
    obs: Observation | None
    reward: float          # 原始 P&L 奖励（评估口径）
    train_reward: float    # 含 hindsight bonus 的训练奖励
    done: bool
    t: int                 # 下一决策点（决策序列内索引）
    pos: float
    cash: float
    vol_label: float       # 当前决策点的波动率标签


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

    @property
    def t(self) -> int:
        return self.market.decision_points[self.step_idx]

    def reset(self) -> Observation:
        self.pos = 0.0
        self.cash = self.cash0
        self.step_idx = 0
        self.n_fills = 0
        return self.market.observe(self.t, self.pos, self.cash, self.cash0)

    def _execute(self, price_off: float, qty: float) -> tuple[float, float]:
        """撮合限价单，返回 (成交数量(带符号), 总交易成本)。

        限价相对对手价：买单以卖一价为基准、卖单以买一价为基准；买向限价 ≥ 卖一价
        （卖向限价 ≤ 买一价）即穿价，按对手一档价成交，否则挂出且不成交。
        成本 = 手续费 + 半价差成本（成交价相对中间价的偏离），
        手续费同步从现金账户扣除，保证奖励与净值口径一致。
        """
        cfg = self.cfg
        m = self.market
        t = self.t
        if qty > 0:
            limit = m.ask1[t] + price_off * cfg.price_tick
        else:
            limit = m.bid1[t] + price_off * cfg.price_tick
        if qty > 0 and limit >= m.ask1[t]:
            fill = min(qty, cfg.max_position - self.pos)
            if fill > 0:
                fee = fill * m.ask1[t] * cfg.fee_rate
                self.cash -= fill * m.ask1[t] + fee
                self.pos += fill
                self.n_fills += 1
                return fill, fee + fill * (m.ask1[t] - m.mid[t])
        elif qty < 0 and limit <= m.bid1[t]:
            fill = min(-qty, self.pos + cfg.max_position)
            if fill > 0:
                fee = fill * m.bid1[t] * cfg.fee_rate
                self.cash += fill * m.bid1[t] - fee
                self.pos -= fill
                self.n_fills += 1
                return -fill, fee + fill * (m.mid[t] - m.bid1[t])
        return 0.0, 0.0

    def step(self, action: tuple[int, int]) -> StepResult:
        cfg = self.cfg
        m = self.market
        t_now = self.t
        off = cfg.price_offsets[action[0]]
        qty = float(cfg.quantities[action[1]])
        _, cost = self._execute(off, qty)
        pos_held = self.pos  # 本决策区间实际持有的仓位（强平前）

        last = self.step_idx == len(m.decision_points) - 1
        t_next = m.n - 1 if last else m.decision_points[self.step_idx + 1]

        # 期末强平（按中间价，计手续费）
        if last and self.pos != 0.0:
            fee = abs(self.pos) * m.mid[t_next] * cfg.fee_rate
            self.cash += self.pos * m.mid[t_next] - fee
            cost += fee
            self.pos = 0.0

        scale = cfg.leverage / self.cash0
        # 论文公式的真实账户口径：r_t = (p_{t+1} − p_t)·pos_t − 交易成本，
        # 成本含手续费与半价差；pos_t 为动作后的持仓
        reward = scale * ((m.mid[t_next] - m.mid[t_now]) * pos_held - cost)
        bonus = 0.0
        if self.hindsight:
            bonus = cfg.hindsight_weight * (m.hindsight_price(t_now) - m.mid[t_now]) * pos_held
        train_reward = reward + scale * bonus

        self.step_idx += 1
        obs = None if last else m.observe(self.t, self.pos, self.cash, self.cash0)
        return StepResult(
            obs=obs,
            reward=float(reward),
            train_reward=float(train_reward),
            done=last,
            t=self.step_idx,
            pos=self.pos,
            cash=self.cash,
            vol_label=m.vol_label(t_now),
        )

    def net_value(self) -> float:
        """杠杆后的净值（以 1 为基准）；权益损失以初始现金为下限（破产吸收态）。"""
        raw = (self.cash + self.market.mid[self.market.n - 1] * self.pos - self.cash0) / self.cash0
        return max(0.0, 1.0 + self.cfg.leverage * raw)
