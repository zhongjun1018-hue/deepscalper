import unittest
from types import SimpleNamespace

import numpy as np

from control.config import Config
from control.env import DayMarket, GridParams, TradingEnv
from strategy.grid import boundaries
from synthetic import synthetic_day


class ScanTest(unittest.TestCase):
    """_scan：自 t+1 起扫描至触发或超时，返回（区间终点, 方向 +1 买 / −1 卖 / 0 超时）。"""

    def make_env(self):
        market = SimpleNamespace(
            n=4,
            bid1=np.array([100.0, 101.0, 100.0, 100.0]),
            ask1=np.array([101.0, 102.0, 101.0, 101.0]),
            scannable=np.array([False, True, True, True]),
        )
        env = object.__new__(TradingEnv)
        env.market = market
        env.cfg = SimpleNamespace(timeout_ticks=3)
        return env

    def test_opposite_best_quote_triggers_fill(self):
        # 买一在 tick 1 上穿卖出线 → 卖出方向 −1
        self.assertEqual(self.make_env()._scan(0, sell=100.5, buy=None), (1, -1))

    def test_ask_crossing_down_triggers_buy(self):
        # 卖一在 tick 2 下穿买入线 → 买入方向 +1
        self.assertEqual(self.make_env()._scan(0, sell=None, buy=101.5), (2, 1))

    def test_no_crossing_times_out(self):
        self.assertEqual(self.make_env()._scan(0, sell=200.0, buy=None), (3, 0))


class CenterAndFlattenTest(unittest.TestCase):
    """平仓按对手一档价全额成交；每笔成交重置中心，超时重置为当拍中间价。

    合成日盘口恒定：买一 9.99 / 卖一 10.01 / 中间价 10.00。
    """

    @classmethod
    def setUpClass(cls):
        cls.market = DayMarket(synthetic_day(), Config())

    def make_env(self) -> TradingEnv:
        return TradingEnv(self.market, hindsight=False)

    def test_flatten_fills_the_whole_excess_at_the_opposite_best_quote(self):
        env = self.make_env()
        env.pos = 60.0   # 超额敞口 +10 手

        env._flatten_to_base(env.t, "flatten")

        fill = env.fills[-1]
        self.assertEqual(fill.kind, "flatten")
        self.assertAlmostEqual(fill.qty, -10.0)      # 一笔全额，不受一档深度限制
        self.assertAlmostEqual(fill.price, 9.99)     # 卖出按买一价
        self.assertAlmostEqual(env.pos, 50.0)
        self.assertAlmostEqual(env.center, 9.99)     # 平仓成交同样重置中心
        self.assertEqual(env.last_fill_tick, env.t)

    def test_flatten_buys_back_at_the_ask_and_is_a_noop_at_base(self):
        env = self.make_env()
        env.pos = 40.0

        env._flatten_to_base(env.t, "flatten")
        self.assertAlmostEqual(env.fills[-1].qty, 10.0)
        self.assertAlmostEqual(env.fills[-1].price, 10.01)   # 买入按卖一价

        n_fills = len(env.fills)
        self.assertEqual(env._flatten_to_base(env.t, "flatten"), 0.0)
        self.assertEqual(len(env.fills), n_fills)    # 净持仓为零：无操作

    def test_interval_without_fill_recenters_to_the_current_mid(self):
        env = self.make_env()
        env.center = 9.5   # 人为偏移中心，验证超时不沿用原中心

        res = env.step(GridParams(width=100.0, size=1))   # 关闭档：区间必然无成交

        self.assertFalse(env.fills)
        self.assertAlmostEqual(env.center, float(self.market.mid[res.t]))

    def test_flatten_gear_is_allowed_only_off_base_position(self):
        env = self.make_env()
        self.assertFalse(env.observation().flatten_allowed)   # 开盘持底仓

        env.pos = 60.0
        env._write_priv(env.t, env.t, width=0.0, size=0)
        self.assertTrue(env.observation().flatten_allowed)


class BoundaryRoundingTest(unittest.TestCase):
    """boundaries：边界取整到最小变动价位，且不缩窄名义半宽。"""

    def test_tick_rounding_does_not_narrow_width(self):
        upper, lower = boundaries(10.0, 0.001, 0.01)

        self.assertAlmostEqual(upper, 10.01)
        self.assertAlmostEqual(lower, 9.99)
        self.assertGreaterEqual(upper - 10.0, 0.001)
        self.assertGreaterEqual(10.0 - lower, 0.001)

    def test_aligned_boundaries_stay_exact(self):
        upper, lower = boundaries(10.0, 0.01, 0.01)

        self.assertAlmostEqual(upper, 10.01)
        self.assertAlmostEqual(lower, 9.99)


if __name__ == "__main__":
    unittest.main()
