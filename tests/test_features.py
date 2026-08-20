import unittest

import numpy as np

from control.config import Config
from control.env import DayMarket
from control.features import (BAR_DIM, MACRO_DIM, MICRO_DIM, MICRO_LOB_DIM,
                              build_macro_features, build_micro_matrix,
                              fit_feature_stats)
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


class FeatureStatsTest(unittest.TestCase):
    """fit_feature_stats：逐标的拟合统计量，行索引即 symbol_id。"""

    def test_rows_are_fitted_per_symbol(self):
        cfg = Config(symbols=("000001", "000002"))
        first = DayMarket(synthetic_day(), cfg, symbol_id=0)
        second = DayMarket(synthetic_day(), cfg, symbol_id=1)
        second.micro = second.micro * 2.0 + 1.0   # 制造跨标的量纲差异
        stats = fit_feature_stats([first, second], cfg)

        self.assertEqual(stats.micro_mean.shape, (2, MICRO_DIM))
        self.assertEqual(stats.macro_mean.shape, (2, MACRO_DIM))
        np.testing.assert_allclose(stats.micro_mean[1],
                                   stats.micro_mean[0] * 2.0 + 1.0, atol=1e-6)
        np.testing.assert_allclose(stats.macro_mean[1], stats.macro_mean[0],
                                   atol=1e-6)


class MacroFeatureTest(unittest.TestCase):
    def test_bar_block_width(self):
        closes = np.linspace(10.0, 10.5, 30)
        bars = np.column_stack([closes - 0.01, closes + 0.02, closes - 0.02,
                                closes, np.ones(30)])
        feats = build_macro_features(bars)

        self.assertEqual(feats.shape, (BAR_DIM,))
        self.assertTrue(np.isfinite(feats).all())
        self.assertAlmostEqual(feats[1], 0.02 / closes[-1], places=6)   # z_high
        self.assertAlmostEqual(feats[-1], 0.0)                          # z_volume


if __name__ == "__main__":
    unittest.main()
