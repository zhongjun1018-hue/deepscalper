import unittest

import numpy as np
import pandas as pd

from data_provider.ticks import (MINUTES_PER_DAY, anchor_ffill, minute_anchors,
                                 minute_index)


class MinuteIndexTest(unittest.TestCase):
    """minute_index：HHMMSSmmm 的 Series → 压缩分钟索引 ndarray（会话边界与网格外）。"""

    def test_session_boundaries(self):
        t = pd.Series(["093000000", "093059000", "093100000", "112959000", "113000000",
                       "130000000", "145600000", "145659000"])
        idx = minute_index(t)
        # 09:30 起每分钟一格；11:30:00 收盘快照归入分钟 119；13:00 起为分钟 120；14:56 为分钟 236
        self.assertIsInstance(idx, np.ndarray)
        self.assertEqual(idx.tolist(), [0, 0, 1, 119, 119, 120, 236, 236])

    def test_out_of_grid(self):
        idx = minute_index(pd.Series(["092959000", "120000000", "145700000"]))
        self.assertEqual(idx.tolist(), [-1, -1, -1])


class MinuteAnchorTest(unittest.TestCase):
    """minute_anchors：每分钟末快照的 tick 索引；anchor_ffill：缺失分钟前向填充。"""

    def test_last_snapshot_per_minute_wins(self):
        frame = pd.DataFrame({"MDTime": ["093000000", "093030000", "093059000",
                                         "093100000", "093300000"]})
        anchors = minute_anchors(frame)
        self.assertEqual(anchors.shape, (MINUTES_PER_DAY,))
        self.assertEqual(anchors[0], 2)     # 分钟 0 的末快照
        self.assertEqual(anchors[1], 3)
        self.assertEqual(anchors[2], -1)    # 分钟 2 无快照
        self.assertEqual(anchors[3], 4)
        self.assertTrue((anchors[4:] == -1).all())

    def test_forward_fill_bridges_missing_minutes(self):
        anchors = np.array([-1, 5, -1, -1, 9])
        np.testing.assert_array_equal(anchor_ffill(anchors), [-1, 5, 5, 5, 9])


if __name__ == "__main__":
    unittest.main()
