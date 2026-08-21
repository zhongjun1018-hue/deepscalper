import unittest

from attribution.features import FEATURE_DICT
from attribution.points import gating_points, replay_enabled
from data_provider.windows import FEATURE_NAMES


def day_fixture() -> dict:
    """10 分钟的最小日 JSON：t0=2，识别信号不利的锚点为分钟 3–5 与 9（锚点 x=m+0.95 落入
    排除段；段末 .9 不含该分钟锚点，与 webviz 导出一致），价格每分钟跌 1。"""
    x = [m + 0.95 for m in range(10)]
    return {
        "t0": 2, "width": 2.0, "x": x, "price": [100.0 - m for m in range(10)],
        "grids": {
            "none": {"events": [
                {"minute": 2, "kind": "open", "exposure": 0},
                {"minute": 5, "kind": "buy", "fill": 95.0, "exposure": 1, "x": 5.5},
            ]},
            "prediction": {
                "excluded": [[3.95, 6.9], [9.95, 9.95]],
                "events": [{"minute": 7, "kind": "open", "exposure": 0}],
            },
        },
    }


class ReplayEnabledTest(unittest.TestCase):
    def test_confirm_n_delays_switch_and_t0_bootstraps(self):
        enabled = replay_enabled(day_fixture(), confirm_n=2)
        # t0=2 放行直接定初态；3 拍不利计 1、4 拍计 2 才停网；6 拍放行计 1、7 拍计 2 才重开；
        # 9 拍不利仅 1 拍不切换
        self.assertEqual(enabled[:10].tolist(),
                         [False, False, True, True, False, False, False, True, True, True])

    def test_open_position_locks_state(self):
        day = day_fixture()
        day["grids"]["prediction"]["events"] = [
            {"minute": 2, "kind": "open", "exposure": 0},
            {"minute": 3, "kind": "buy", "fill": 96.0, "exposure": 1},
        ]
        enabled = replay_enabled(day, confirm_n=2)
        self.assertTrue(enabled[2:].all())   # 带仓期间不读信号，全程不停网


class FeatureDictTest(unittest.TestCase):
    def test_covers_feature_names(self):
        self.assertEqual(set(FEATURE_DICT), set(FEATURE_NAMES))


class GatingPointsTest(unittest.TestCase):
    def test_points_and_counterfactual(self):
        points = gating_points(day_fixture(), confirm_n=2)
        self.assertEqual(len(points), 1)
        point = points[0]
        self.assertEqual((point["start"], point["end"], point["minutes"]), (4, 6, 3))
        self.assertAlmostEqual(point["drift_W"], (94.0 - 96.0) / 2.0)
        always_on = point["always_on"]
        self.assertEqual((always_on["buys"], always_on["exposure_in"], always_on["exposure_out"]),
                         (1, 0, 1))
        self.assertAlmostEqual(always_on["mtm_W"], (94.0 - 95.0) / 2.0)   # 买 95、段末 94
        self.assertEqual(point["gated"]["fills"], [])


if __name__ == "__main__":
    unittest.main()
