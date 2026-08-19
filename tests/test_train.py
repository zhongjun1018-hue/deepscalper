import unittest

import numpy as np

from data_provider.windows import FEATURE_NAMES

from control.config import Config
from control.features import MACRO_FEATURE_COLUMNS, PRED_DIM, WINDOW_DIM
from control.train import build_markets, eval_days
from synthetic import synthetic_day

ROWS = 7   # 合成缓存的样本行数


class EvalDaysTest(unittest.TestCase):
    def test_checkpoints_split_days_evenly_and_end_at_last_day(self):
        self.assertEqual(eval_days(68, 3), {21, 44, 67})
        self.assertEqual(eval_days(9, 1), {8})
        self.assertEqual(eval_days(2, 3), {0, 1})   # 评估点多于交易日时逐日评估


class BuildMarketsTest(unittest.TestCase):
    """build_markets：统一缓存的当日行块按宏观列选取后拼为 R×(24+5)，缺当日行时补零块。"""

    def make_cache(self, date):
        # 第 j 列填 j，使「是否只取了宏观列」可直接比对
        features = np.tile(np.arange(len(FEATURE_NAMES), dtype=np.float32), (1, ROWS, 1))
        return {
            "dates": np.array([date]),
            "features": features,
            "preds": np.full((1, ROWS, PRED_DIM), 3.0, dtype=np.float32),
        }

    def build(self, cache):
        markets = build_markets([synthetic_day()], Config(), cache)
        self.assertEqual(len(markets), 1)   # 合成日可交易（ATR 有效且长度足够）
        return markets[0]

    def test_present_day_takes_only_the_macro_columns(self):
        market = self.build(self.make_cache("20260102"))

        self.assertEqual(market.window.shape, (ROWS, WINDOW_DIM + PRED_DIM))
        np.testing.assert_array_equal(market.window[:, :WINDOW_DIM],
                                      np.tile(MACRO_FEATURE_COLUMNS, (ROWS, 1)))
        np.testing.assert_array_equal(market.window[:, WINDOW_DIM:], 3.0)

    def test_missing_day_is_zero_filled(self):
        market = self.build(self.make_cache("19990101"))

        np.testing.assert_array_equal(market.window, 0.0)

    def test_absent_cache_falls_back_to_zero_block(self):
        market = self.build(None)

        self.assertEqual(market.window.shape[1], WINDOW_DIM + PRED_DIM)
        np.testing.assert_array_equal(market.window, 0.0)


if __name__ == "__main__":
    unittest.main()
