import unittest

import numpy as np
import pandas as pd

from data_provider.ticks import minute_index


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


if __name__ == "__main__":
    unittest.main()
