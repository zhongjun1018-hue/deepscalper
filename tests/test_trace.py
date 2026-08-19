import unittest

from data_provider.ticks import load_days
from control.config import Config
from control.env import DayMarket
from control.trace import trace_day


class TraceDayTest(unittest.TestCase):
    """trace_day：真实交易日 + 零块窗口的 DayMarket 上回放固定半宽网格策略。

    固定策略不经过网络，policy 直接返回档位索引（0.1×ATR 半宽档、数量档 1）。
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()
        day = load_days("301308")[3]   # 第 4 日：ATR 首个有效日
        cls.market = DayMarket(day, cls.cfg)   # window 缺省为零块
        cls.result = trace_day(cls.market, cls.fixed_policy)

    @classmethod
    def fixed_policy(cls, obs):
        return (cls.cfg.widths.index(0.1), 0)

    def test_decision_fields(self):
        decisions = self.result["decisions"]

        self.assertTrue(decisions)
        ticks = []
        for d in decisions:
            self.assertEqual(set(d), {"t", "width", "size", "center", "upper", "lower"})
            self.assertEqual(d["width"], 0.1)   # 固定半宽档
            self.assertEqual(d["size"], 1)
            self.assertGreater(d["upper"], d["center"])
            self.assertGreater(d["center"], d["lower"])
            ticks.append(d["t"])
        self.assertTrue(all(later > earlier for earlier, later in zip(ticks, ticks[1:])))

    def test_fill_fields_and_conservation(self):
        fills = self.result["fills"]

        self.assertTrue(fills)
        signed = 0.0
        for f in fills:
            self.assertEqual(set(f), {"t", "side", "price", "qty"})
            self.assertIn(f["side"], ("buy", "sell"))
            self.assertGreater(f["price"], 0.0)
            self.assertGreater(f["qty"], 0.0)
            signed += f["qty"] if f["side"] == "buy" else -f["qty"]
        # 数量守恒：开盘持底仓、日终平回底仓，全日带符号成交量合计为 0
        self.assertAlmostEqual(signed, 0.0, delta=1e-9)

    def test_log_matches_the_recorded_events(self):
        log = self.result["log"]

        self.assertEqual(log["n_decisions"], len(self.result["decisions"]))
        # env.fills 只含网格成交；fills 另含日终扫单补记，不少于前者
        self.assertGreaterEqual(len(self.result["fills"]), log["n_fills"])
        self.assertEqual(log["n_buys"] + log["n_sells"], log["n_fills"])


class TraceFlattenTest(unittest.TestCase):
    """平仓档（width == 0）的扫单补记：偏离底仓即平仓，fills 与环境记账守恒。"""

    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()
        day = load_days("301308")[3]
        cls.result = trace_day(DayMarket(day, cls.cfg), cls.flatten_policy)

    @classmethod
    def flatten_policy(cls, obs):
        # 私有序列首列为 pos/max_position，底仓对应 0.5；偏离即在决策点平回底仓
        if obs.private[-1, 0] != 0.5:
            return (0, 0)
        return (cls.cfg.widths.index(0.05), 0)   # 最密网格，尽快产生超额敞口

    def test_flatten_sweep_is_recorded(self):
        decisions = self.result["decisions"]
        fills = self.result["fills"]

        flatten_ticks = {d["t"] for d in decisions if d["width"] == 0.0}
        self.assertTrue(flatten_ticks)
        for d in decisions:
            if d["width"] == 0.0:   # 平仓档不建网格，生效数量记 0
                self.assertIsNone(d["upper"])
                self.assertIsNone(d["lower"])
                self.assertEqual(d["size"], 0)
        sweep_fills = [f for f in fills if f["t"] in flatten_ticks]
        self.assertTrue(sweep_fills)   # 平仓扫单按逐档均价补记在决策点上
        signed = sum(f["qty"] if f["side"] == "buy" else -f["qty"] for f in fills)
        self.assertAlmostEqual(signed, 0.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
