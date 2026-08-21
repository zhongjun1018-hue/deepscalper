import unittest

import numpy as np

from strategy.metrics import closure_rate, financial_metrics, summarize


class FinancialMetricsTest(unittest.TestCase):
    def test_sortino_uses_zero_target_downside_deviation(self):
        metrics = financial_metrics(np.array([-0.01, -0.01]))

        self.assertAlmostEqual(metrics["SoR"], -1.0)

    def test_empty_returns_are_zero(self):
        self.assertEqual(financial_metrics(np.array([])),
                         {"TR": 0.0, "SR": 0.0, "CR": 0.0, "SoR": 0.0})

    def test_first_day_drawdown_is_counted(self):
        metrics = financial_metrics(np.array([-0.1, 0.0]))

        self.assertAlmostEqual(metrics["TR"], -0.1)
        self.assertAlmostEqual(metrics["CR"], -0.05 / 0.1)


class ClosureRateTest(unittest.TestCase):
    def test_pairing_ratio_of_one_sided_and_balanced_days(self):
        self.assertAlmostEqual(closure_rate(3, 1), 0.5)
        self.assertAlmostEqual(closure_rate(4, 4), 1.0)
        self.assertEqual(closure_rate(0, 0), 0.0)


class SummarizeTest(unittest.TestCase):
    """summarize 的字段契约（统一回测各模式共用同一逐日记录口径）。"""

    records = [
        {"g": 3.0, "n_buys": 2, "n_sells": 2, "closure_rate": 1.0,
         "width_rel": 0.006},
        {"g": -1.0, "n_buys": 3, "n_sells": 1,
         "closure_rate": 0.5, "width_rel": float("nan")},   # 无网格触发时间的日
        {"g": 0.0, "n_buys": 0, "n_sells": 0,
         "closure_rate": 0.0, "width_rel": float("nan")},   # 零交易日
    ]

    def test_daily_means(self):
        summary = summarize(self.records)

        self.assertEqual(summary["n_days"], 3)
        # g 的矩按有值日计（含零交易日）
        self.assertAlmostEqual(summary["mean_g"], 2.0 / 3.0)
        # 闭环率与笔数的分母只数该指标非零的日
        self.assertAlmostEqual(summary["mean_closure_rate"], 0.75)
        self.assertAlmostEqual(summary["mean_trades"], 4.0)
        self.assertAlmostEqual(summary["mean_buys"], 2.5)
        self.assertAlmostEqual(summary["mean_sells"], 1.5)
        # 宽幅的矩只计有值的交易日
        self.assertAlmostEqual(summary["mean_width_rel"], 0.006)

    def test_all_zero_days_yield_nan_counts(self):
        summary = summarize([{"g": 0.0, "n_buys": 0, "n_sells": 0,
                              "closure_rate": 0.0, "width_rel": float("nan")}])

        self.assertTrue(np.isnan(summary["mean_closure_rate"]))
        self.assertTrue(np.isnan(summary["mean_trades"]))

    def test_field_contract(self):
        summary = summarize(self.records)

        for key in ("n_days", "mean_g", "std_g", "mean_closure_rate", "mean_trades",
                    "mean_buys", "mean_sells", "mean_width_rel"):
            self.assertIn(key, summary)

    def test_empty_records(self):
        summary = summarize([])

        self.assertEqual(summary["n_days"], 0)
        self.assertTrue(np.isnan(summary["mean_g"]))
        self.assertTrue(np.isnan(summary["mean_trades"]))
        self.assertTrue(np.isnan(summary["mean_closure_rate"]))


if __name__ == "__main__":
    unittest.main()
