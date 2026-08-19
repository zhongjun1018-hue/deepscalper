import unittest

import numpy as np

from data_provider.windows import TARGET_NAMES
from forecast.signals import GATES, SCHEMES, build_gate_masks

_INDEX = {name: k for k, name in enumerate(TARGET_NAMES)}
TICKS = 8


def make_sources(days, ticks, columns):
    """构造 (D,T,5) 的 oracle / prediction 信号数组，未指定 tick 记 NaN（不可判定）。

    columns: {目标名: (oracle 值序列, prediction 值序列)}，与 ticks 逐项对应。
    """
    sources = {scheme: np.full((len(days), TICKS, len(TARGET_NAMES)), np.nan)
               for scheme in ("oracle", "prediction")}
    for name, (oracle_values, prediction_values) in columns.items():
        for d in range(len(days)):
            for k, tick in enumerate(ticks):
                sources["oracle"][d, tick, _INDEX[name]] = oracle_values[k]
                sources["prediction"][d, tick, _INDEX[name]] = prediction_values[k]
    return sources


class GateMaskTest(unittest.TestCase):
    """掩码逐 tick 判定：True 表示该 tick 的信号排除开新网，两个 scheme 各自独立成掩码。"""

    def test_each_scheme_marks_its_own_hits(self):
        sources = make_sources(["20260102"], [2, 3, 4], {
            "resid_abs_q90": ([0.002, 0.002, 0.002], [0.014, 0.002, 0.002]),
            "abs_slope": ([0.014, 0.001, 0.009], [0.014, 0.014, 0.009]),
        })
        anchors = np.full((1, TICKS), 10.0)  # w = 0.1 / 10 = 0.01

        masks, gated = build_gate_masks(sources, np.array(["20260102"]),
                                        np.array([0.1]), anchors)

        oracle = np.zeros(TICKS, dtype=bool)
        oracle[2] = True    # 残差 0.2 < 1.3 且斜率 1.4 > 0.9
        prediction = np.zeros(TICKS, dtype=bool)
        prediction[3] = True
        np.testing.assert_array_equal(masks[("residual", "oracle", "20260102")], oracle)
        np.testing.assert_array_equal(masks[("residual", "prediction", "20260102")], prediction)
        self.assertEqual(gated, {"residual": {"total": 3, "oracle": 1, "prediction": 1}})

    def test_trend_gate_scales_slope_by_relative_grid_width(self):
        # 两日信号相同，仅当日半宽不同（w = 0.01 / 0.02）：斜率宽比 0.91 只在第一日 > 0.9
        sources = make_sources(["20260102", "20260103"], [2], {
            "resid_abs_q90": ([0.001], [0.001]),
            "abs_slope": ([0.0091], [0.0091]),
        })
        anchors = np.full((2, TICKS), 10.0)

        masks, gated = build_gate_masks(
            sources, np.array(["20260102", "20260103"]), np.array([0.1, 0.2]), anchors)

        expected = np.zeros(TICKS, dtype=bool)
        expected[2] = True
        np.testing.assert_array_equal(masks[("residual", "prediction", "20260102")], expected)
        self.assertNotIn(("residual", "prediction", "20260103"), masks)
        self.assertEqual(gated["residual"], {"total": 2, "oracle": 1, "prediction": 1})

    def test_three_schemes_and_none_never_masks(self):
        self.assertEqual(SCHEMES, ["none", "oracle", "prediction"])
        self.assertEqual(GATES, ["residual"])

        sources = make_sources(["20260102"], [2], {
            "resid_abs_q90": ([0.001], [0.001]),
            "abs_slope": ([0.02], [0.02]),
        })
        anchors = np.full((1, TICKS), 10.0)
        masks, _ = build_gate_masks(sources, np.array(["20260102"]), np.array([0.1]), anchors)
        # none 为常开基线，永不产生排除掩码
        self.assertTrue(masks)
        self.assertEqual({scheme for _, scheme, _ in masks}, {"oracle", "prediction"})


if __name__ == "__main__":
    unittest.main()
