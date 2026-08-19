import unittest

import numpy as np

from strategy.engine import run_day


class StrictCrossingTest(unittest.TestCase):
    """严格穿价成交：对手方一档恰好触及边界（等于边界价）不成交。"""

    def test_touching_the_boundary_does_not_fill(self):
        # width=0.1，开网中心 10（上界 10.1 / 下界 9.9）
        result = run_day(
            bid1=np.array([10.0, 10.1, 10.1001, 10.2]),
            ask1=np.array([10.0, 10.1, 10.1, 9.8998]),
            mid=np.full(4, 10.0),
            hard_exclude=None, width=0.1,
        )
        # tick1 买一 10.1 触上界不成交；tick2 10.1001 上穿成交（卖 10.1，中心移 10.1）；
        # tick3 买一 10.2 恰触新上界不成交，卖一 9.8998 下穿下界 10.0 成交（买 10.0）
        self.assertEqual((result["buys"], result["sells"]), (1, 1))
        self.assertAlmostEqual(result["grid_profit"], 1.0)
        self.assertAlmostEqual(result["grid_profit_lower"], 1.0)

    def test_trace_events_feed_webviz(self):
        result = run_day(
            bid1=np.array([10.0, 10.0, 10.0]),
            ask1=np.array([10.0, 9.8, 9.9]),
            mid=np.full(3, 10.0),
            hard_exclude=None, width=0.1, trace=True,
        )
        opening, fill = result["events"]
        self.assertEqual(opening["kind"], "open")
        for key in ("index", "center", "upper", "lower", "exposure"):
            self.assertIn(key, opening)
        self.assertEqual(fill["kind"], "buy")
        self.assertAlmostEqual(fill["fill"], 9.9)  # 成交价为触发边界
        self.assertAlmostEqual(fill["center"], 9.9)


class GridProfitTest(unittest.TestCase):
    """无量纲 grid_profit：0.5×(成交笔数+敞口²)+敞口浮动项；下界为 0.5×(笔数−敞口²)−|敞口|。"""

    def test_dimensionless_profit_and_lower_bound(self):
        result = run_day(
            bid1=np.array([10.0, 10.0, 10.0]),
            ask1=np.array([10.0, 9.8, 9.9]),
            mid=np.full(3, 10.0),
            hard_exclude=None, width=0.1,
        )
        # 买入 1 笔后敞口 1 持到日终，收盘价等于开网中心（浮动项为 0）
        self.assertEqual((result["buys"], result["sells"]), (1, 0))
        self.assertAlmostEqual(result["grid_profit"], 1.0)
        self.assertAlmostEqual(result["grid_profit_lower"], -1.0)
        # 下界与实现值的关系：浮动项非负时实现值不低于下界
        self.assertGreaterEqual(result["grid_profit"], result["grid_profit_lower"])


class GateTimingTest(unittest.TestCase):
    """门控只在净持仓为 0 时按当拍信号生效：持仓途中的信号变化不生效，直至敞口归零。"""

    def test_gate_waits_for_flat_position(self):
        bid1 = np.array([10.0, 10.0, 10.0, 10.05, 10.0])
        ask1 = np.array([10.0, 9.8, 10.0, 10.0, 9.8])
        mid = np.full(5, 10.0)
        excluded = np.array([False, False, True, True, True])

        gated = run_day(bid1, ask1, mid, hard_exclude=excluded, width=0.1)
        # tick1 买入后敞口 1 带入被排除的 tick：门控不生效，tick3 上穿仍成交（卖 10.0）；
        # 敞口归零后立即按 tick3 的信号收网，tick4 的下穿不再开新仓
        self.assertEqual((gated["buys"], gated["sells"]), (1, 1))

        open_day = run_day(bid1, ask1, mid, hard_exclude=None, width=0.1)
        self.assertEqual((open_day["buys"], open_day["sells"]), (2, 1))

    def test_gate_blocks_reopen_when_flat(self):
        bid1 = np.array([10.0, 10.2, 10.2])
        ask1 = np.array([10.0, 10.0, 10.0])
        mid = np.full(3, 10.0)
        excluded = np.array([False, True, True])

        gated = run_day(bid1, ask1, mid, hard_exclude=excluded, width=0.1)
        self.assertEqual((gated["buys"], gated["sells"]), (0, 0))

        open_day = run_day(bid1, ask1, mid, hard_exclude=None, width=0.1)
        self.assertEqual((open_day["buys"], open_day["sells"]), (0, 1))

    def test_flat_position_rechecks_on_the_interval(self):
        """空仓期间每 decide_interval 个 tick 复判一次：两次判定之间的信号变化不生效。"""
        bid1 = np.full(5, 10.0)
        ask1 = np.array([10.0, 10.0, 10.0, 10.0, 9.8])
        mid = np.full(5, 10.0)
        # tick0 放行开网，tick1 命中但落在两次判定之间，tick4 的下穿因此仍然成交
        excluded = np.array([False, True, False, False, False])

        spaced = run_day(bid1, ask1, mid, hard_exclude=excluded, width=0.1, decide_interval=4)
        self.assertEqual((spaced["buys"], spaced["sells"]), (1, 0))

    def test_returning_to_flat_rejudges_immediately(self):
        """成交后敞口归零即刻按当拍信号重判，不等复判间隔到期。"""
        bid1 = np.array([10.0, 10.0, 10.05, 10.0])
        ask1 = np.array([10.0, 9.8, 10.0, 9.8])
        mid = np.full(4, 10.0)
        # 间隔设得比全天还长：tick2 的排除只可能经「敞口归零即刻重判」生效
        excluded = np.array([False, False, True, False])

        gated = run_day(bid1, ask1, mid, hard_exclude=excluded, width=0.1, decide_interval=100)
        self.assertEqual((gated["buys"], gated["sells"]), (1, 1))

        open_day = run_day(bid1, ask1, mid, hard_exclude=None, width=0.1)
        self.assertEqual((open_day["buys"], open_day["sells"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
