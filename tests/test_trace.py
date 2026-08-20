import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from data_provider.ticks import load_days
from data_provider.windows import WindowSpec
from control.config import Config
from control.env import DayMarket, Observation
from control.features import MACRO_DIM, MICRO_DIM, PRIVATE_DIM, FeatureStats
from control.model import BranchQNetwork
from control.trace import greedy_policy, load_checkpoint, trace_day
from control.train import save_checkpoint


class CheckpointRoundTripTest(unittest.TestCase):
    """save_checkpoint → load_checkpoint 往返：权重一致，嵌套 WindowSpec 还原为 dataclass。"""

    def test_config_weights_and_stats_survive_the_roundtrip(self):
        cfg = Config(symbols=("301308",))
        net = BranchQNetwork(cfg)
        stats = FeatureStats(np.zeros((1, 66)), np.ones((1, 66)),
                             np.zeros((1, 40)), np.ones((1, 40)),
                             clip=cfg.norm_clip)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "GRID_w0.1_lam3_seed0.pt")
            save_checkpoint(SimpleNamespace(online=net), cfg, stats, path)
            restored, loaded_cfg, loaded_stats = load_checkpoint(path, torch.device("cpu"))

        self.assertIsInstance(loaded_cfg.window, WindowSpec)
        self.assertEqual(loaded_cfg, cfg)
        np.testing.assert_array_equal(loaded_stats.macro_std, stats.macro_std)
        self.assertEqual(loaded_stats.clip, cfg.norm_clip)
        for name, weight in net.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], weight)


class GreedyPolicyTest(unittest.TestCase):
    """greedy_policy：网络前向取各分支 argmax 档位。"""

    def make_obs(self, cfg: Config) -> Observation:
        return Observation(
            micro_lob=np.zeros((cfg.micro_steps, MICRO_DIM), dtype=np.float32),
            private=np.zeros((cfg.micro_steps, PRIVATE_DIM), dtype=np.float32),
            macro=np.zeros(MACRO_DIM, dtype=np.float32))

    def test_gears_are_valid(self):
        cfg = Config(symbols=("301308",))
        net = BranchQNetwork(cfg).eval()
        obs = self.make_obs(cfg)
        device = torch.device("cpu")

        width_gear, size_gear = greedy_policy(net, device)(obs)
        self.assertIn(width_gear, range(cfg.n_width))
        self.assertIn(size_gear, range(cfg.n_size))

    def test_flatten_gear_is_masked_when_net_position_is_zero(self):
        cfg = Config(symbols=("301308",))
        net = BranchQNetwork(cfg).eval()
        obs = self.make_obs(cfg)
        obs.flatten_allowed = False

        # 平仓档（半宽档 0）在净持仓为零时不可选（与 BranchQAgent.greedy 同一掩码）
        self.assertNotEqual(greedy_policy(net, torch.device("cpu"))(obs)[0], 0)


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
            self.assertEqual(set(f), {"t", "side", "price", "qty", "kind"})
            self.assertIn(f["side"], ("buy", "sell"))
            self.assertIn(f["kind"], ("grid", "immediate", "flatten", "liquidate"))
            self.assertGreater(f["price"], 0.0)
            self.assertGreater(f["qty"], 0.0)
            signed += f["qty"] if f["side"] == "buy" else -f["qty"]
        # 数量守恒：开盘持底仓、日终平回底仓，全日带符号成交量合计为 0
        self.assertAlmostEqual(signed, 0.0, delta=1e-9)

    def test_log_matches_the_recorded_events(self):
        log, fills = self.result["log"], self.result["fills"]

        self.assertEqual(log["n_decisions"], len(self.result["decisions"]))
        # fills 含全部成交，log 的成交口径只计日内主动成交（7.4）
        liquidate = [f for f in fills if f["kind"] == "liquidate"]
        self.assertEqual(len(fills) - len(liquidate), log["n_fills"])
        self.assertEqual(log["n_buys"] + log["n_sells"], log["n_fills"])
        self.assertAlmostEqual(log["liquidated_lots"],
                               sum(f["qty"] for f in liquidate), delta=1e-9)


class TraceFlattenTest(unittest.TestCase):
    """平仓档（width == 0）：偏离底仓即按对手一档价平仓，成交守恒且计入日内成交口径。"""

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

    def test_flatten_fill_is_recorded(self):
        decisions = self.result["decisions"]
        fills = self.result["fills"]

        flatten_ticks = {d["t"] for d in decisions if d["width"] == 0.0}
        self.assertTrue(flatten_ticks)
        for d in decisions:
            if d["width"] == 0.0:   # 平仓档不建网格，生效数量记 0
                self.assertIsNone(d["upper"])
                self.assertIsNone(d["lower"])
                self.assertEqual(d["size"], 0)
        flatten = [f for f in fills if f["kind"] == "flatten"]
        self.assertTrue(flatten)   # 平仓按对手一档价记在决策点上
        self.assertTrue(all(f["t"] in flatten_ticks for f in flatten))
        signed = sum(f["qty"] if f["side"] == "buy" else -f["qty"] for f in fills)
        self.assertAlmostEqual(signed, 0.0, delta=1e-9)

    def test_intraday_flatten_counts_but_day_end_liquidation_does_not(self):
        log = self.result["log"]
        flatten = [f for f in self.result["fills"] if f["kind"] == "flatten"]

        # 日内平仓是主动成交，计入买卖笔数与闭环率；日终清仓只报告手数（7.4）
        self.assertEqual(log["n_flatten"], len(flatten))
        self.assertAlmostEqual(log["flattened_lots"],
                               sum(f["qty"] for f in flatten), delta=1e-9)
        grid = [f for f in self.result["fills"] if f["kind"] in ("grid", "immediate")]
        self.assertEqual(log["n_fills"], len(grid) + len(flatten))
        self.assertGreater(log["closure_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
