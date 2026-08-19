import unittest

import numpy as np

from control.features import (BAR_DIM, MICRO_DIM, MICRO_LOB_DIM, build_macro_features,
                              build_micro_matrix)
from synthetic import synthetic_day


class MicroMatrixTest(unittest.TestCase):
    """build_micro_matrix：逐快照 66 维，前 40 维十档相对价量、后 26 维微观结构。"""

    def setUp(self):
        self.frame = synthetic_day().frame
        self.micro = build_micro_matrix(self.frame)

    def test_shape_and_finiteness(self):
        self.assertEqual(self.micro.shape, (len(self.frame), MICRO_DIM))
        self.assertEqual(self.micro.dtype, np.float32)
        self.assertTrue(np.isfinite(self.micro).all())

    def test_level_one_reference_columns(self):
        # 买一价相对买一价恒为 0；买一 / 卖一量相对各自一档量取 log1p(1)
        np.testing.assert_allclose(self.micro[:, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(self.micro[:, 20], np.log(2.0), atol=1e-6)
        np.testing.assert_allclose(self.micro[:, 30], np.log(2.0), atol=1e-6)

    def test_intraday_state_matches_window_definition(self):
        # 后 26 维的末 5 维：日内开盘偏离、距涨停、距跌停、区间位置、时段
        log_mid, day = np.log(10.0), self.micro[:, MICRO_LOB_DIM:]
        np.testing.assert_allclose(day[:, -5], log_mid - np.log(10.0), atol=1e-6)
        np.testing.assert_allclose(day[:, -4], np.log(11.0) - log_mid, atol=1e-6)
        np.testing.assert_allclose(day[:, -3], log_mid - np.log(9.0), atol=1e-6)
        np.testing.assert_allclose(day[:, -2], (10.0 - 9.8) / (10.2 - 9.8), atol=1e-6)
        np.testing.assert_allclose(day[:, -1], 0.0)   # 合成日全在上午时段

    def test_quote_change_and_queue_churn_are_zero_on_a_frozen_book(self):
        # 盘口恒定：报价变化恒 0，一档队列变化率恒 0
        book = self.micro[:, MICRO_LOB_DIM:]
        np.testing.assert_allclose(book[:, 9], 0.0)
        np.testing.assert_allclose(book[:, 10], 0.0)


class MacroFeatureTest(unittest.TestCase):
    def test_bar_block_width(self):
        mid = np.linspace(10.0, 10.5, 600)
        volume = np.ones(600)
        bars = build_macro_features(mid, volume, n_bars=30)

        self.assertEqual(bars.shape, (BAR_DIM,))
        self.assertTrue(np.isfinite(bars).all())


if __name__ == "__main__":
    unittest.main()
