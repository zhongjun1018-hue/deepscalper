import unittest

import numpy as np

from data_provider.windows import TARGET_NAMES
from forecast.regime.classify import calibrate_threshold
from forecast.regime.config import RegimeConfig
from forecast.regime.data import RESID_IDX, SLOPE_IDX, SymbolBank
from forecast.regime.labels import pattern_labels, rule_hits

N_TICKS = 21   # 合成 bank 的分钟行数（真实缓存为 237）


def make_bank(resid, slope, judgeable=None):
    """单日合成 bank：realized 只填 resid/slope 两列（其余列置 1，恒有限）。"""
    realized = np.ones((1, N_TICKS, len(TARGET_NAMES)))
    realized[0, :, RESID_IDX] = resid
    realized[0, :, SLOPE_IDX] = slope
    if judgeable is None:
        judgeable = np.ones((1, N_TICKS), dtype=bool)
    realized[~judgeable] = np.nan
    return SymbolBank(symbol="TEST", dates=np.array(["20260102"]), split=None,
                      width=np.array([0.1]), features=np.zeros((1, N_TICKS, 47)),
                      realized=realized, judgeable=judgeable,
                      anchors=np.tile(np.arange(N_TICKS), (1, 1)), quotes={},
                      open_px={})


def make_config(**overrides):
    defaults = dict(sticky_stay=0.99, emission_noise=0.3)
    defaults.update(overrides)
    return RegimeConfig(**defaults)


class RuleHitsTest(unittest.TestCase):
    """规则观测：低残差宽比 ∧ 高斜率宽比 记 1，不可判定记 -1。"""

    def test_marks_low_resid_strong_slope(self):
        resid = np.full(N_TICKS, 2.0)
        slope = np.full(N_TICKS, 0.5)
        resid[3], slope[3] = 0.2, 1.4    # 命中：残差 0.2 < 1.3 且斜率 1.4 > 0.9
        slope[4] = 1.4                   # 未命中：残差 2.0 不低
        hits = rule_hits(make_bank(resid, slope), make_config())

        expected = np.zeros(N_TICKS, dtype=np.int8)
        expected[3] = 1
        np.testing.assert_array_equal(hits[0], expected)

    def test_unjudgeable_ticks_are_minus_one(self):
        judgeable = np.ones((1, N_TICKS), dtype=bool)
        judgeable[0, :5] = False
        hits = rule_hits(make_bank(np.full(N_TICKS, 0.2), np.full(N_TICKS, 1.4),
                                   judgeable), make_config())
        np.testing.assert_array_equal(hits[0, :5], -1)
        np.testing.assert_array_equal(hits[0, 5:], 1)


class PatternLabelsTest(unittest.TestCase):
    """事后标签：粘性平滑吸收孤立翻转、保留连片段，不可判定处记 -1。"""

    def test_isolated_flip_is_absorbed(self):
        resid = np.full(N_TICKS, 2.0)
        slope = np.full(N_TICKS, 0.5)
        resid[10], slope[10] = 0.2, 1.4    # 全 0 序列中单拍命中
        labels = pattern_labels(make_bank(resid, slope), make_config())
        np.testing.assert_array_equal(labels[0], np.zeros(N_TICKS, dtype=np.int8))

    def test_contiguous_run_is_kept(self):
        resid = np.full(N_TICKS, 2.0)
        slope = np.full(N_TICKS, 0.5)
        resid[8:], slope[8:] = 0.2, 1.4    # 后段连片命中
        labels = pattern_labels(make_bank(resid, slope), make_config())
        self.assertEqual(labels[0, 2], 0)
        np.testing.assert_array_equal(labels[0, 12:], 1)

    def test_unjudgeable_ticks_are_minus_one(self):
        judgeable = np.ones((1, N_TICKS), dtype=bool)
        judgeable[0, -4:] = False
        labels = pattern_labels(make_bank(np.full(N_TICKS, 0.2),
                                          np.full(N_TICKS, 1.4), judgeable),
                                make_config())
        np.testing.assert_array_equal(labels[0, -4:], -1)
        np.testing.assert_array_equal(labels[0, :-4], 1)


class CalibrateThresholdTest(unittest.TestCase):
    """率配平：阈值使预测判正率等于真值不利占比。"""

    def test_positive_rate_matches_base_rate(self):
        rng = np.random.default_rng(0)
        prob = rng.uniform(size=1000)
        labels = (rng.uniform(size=1000) < 0.3).astype(np.int8)
        threshold = calibrate_threshold(prob, labels)
        self.assertAlmostEqual(float((prob > threshold).mean()),
                               float(labels.mean()), delta=0.01)


if __name__ == "__main__":
    unittest.main()
