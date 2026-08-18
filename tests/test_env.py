import unittest
from types import SimpleNamespace

import numpy as np

from gridscalper.env import TradingEnv, _to_tick


class MatchingTest(unittest.TestCase):
    def test_opposite_best_quote_triggers_fill(self):
        market = SimpleNamespace(
            n=4,
            bid_p=np.array([[100.0], [101.0], [100.0], [100.0]]),
            ask_p=np.array([[101.0], [102.0], [101.0], [101.0]]),
            scannable=np.array([False, True, True, True]),
        )
        env = object.__new__(TradingEnv)
        env.market = market
        env.cfg = SimpleNamespace(timeout_ticks=3)

        self.assertEqual(env._scan(0, sell=100.5, buy=None), (1, -1))

    def test_tick_rounding_does_not_narrow_width(self):
        self.assertEqual(_to_tick(10.001, 0.01, up=True), 10.01)
        self.assertEqual(_to_tick(9.999, 0.01, up=False), 9.99)


if __name__ == "__main__":
    unittest.main()
