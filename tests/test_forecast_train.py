import unittest

import numpy as np

from data_provider.windows import FEATURE_NAMES, TARGET_NAMES
from forecast.train import symbol_rows, training_rows

ROWS = 4


def make_bank(dates, value):
    """合成单标的缓存条目：特征与目标全部填 value，便于比对行来源。"""
    days = len(dates)
    return {
        "dates": np.array(dates),
        "features": np.full((days, ROWS, len(FEATURE_NAMES)), value, dtype=np.float32),
        "targets": np.full((days, ROWS, len(TARGET_NAMES)), value, dtype=np.float32),
    }


class SymbolRowsTest(unittest.TestCase):
    """行过滤口径：5 目标全有限 ∧ 任一窗口特征有限；日期集合之外的日不取行。"""

    def test_invalid_rows_and_foreign_dates_are_dropped(self):
        bank = make_bank(["20260102", "20260103"], 1.0)
        bank["targets"][0, 0, 0] = np.nan     # 目标缺失的行剔除
        bank["features"][0, 1] = np.nan       # 特征全缺失的行剔除

        feats, targs = symbol_rows(bank, {"20260102"}, stride=1)

        self.assertEqual((len(feats), len(targs)), (ROWS - 2, ROWS - 2))
        np.testing.assert_array_equal(targs, 1.0)


class TrainingRowsTest(unittest.TestCase):
    """跨标的池化组装：symbol_id 为排序后标的集合中的索引（模型的分类特征口径）。"""

    def test_symbol_ids_follow_sorted_order(self):
        banks = {"B": make_bank(["20260102"], 2.0), "A": make_bank(["20260102"], 1.0)}
        date_set = {"A": {"20260102"}, "B": {"20260102"}}

        x, y = training_rows(banks, date_set, stride=1)

        self.assertEqual(x.shape, (2 * ROWS, len(FEATURE_NAMES) + 1))
        np.testing.assert_array_equal(x[:ROWS, -1], 0.0)   # A 排序在前
        np.testing.assert_array_equal(x[ROWS:, -1], 1.0)
        np.testing.assert_array_equal(x[:ROWS, 0], 1.0)
        np.testing.assert_array_equal(y[ROWS:], 2.0)


if __name__ == "__main__":
    unittest.main()
