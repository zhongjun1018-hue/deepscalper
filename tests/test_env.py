import unittest
from types import SimpleNamespace

import numpy as np

from control.env import TradingEnv
from strategy.grid import boundaries


class ScanTest(unittest.TestCase):
    """_scan：自 t+1 起扫描至触发或超时，返回（区间终点, 方向 +1 买 / −1 卖 / 0 超时）。"""

    def make_env(self):
        market = SimpleNamespace(
            n=4,
            bid_p=np.array([[100.0], [101.0], [100.0], [100.0]]),
            ask_p=np.array([[101.0], [102.0], [101.0], [101.0]]),
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
