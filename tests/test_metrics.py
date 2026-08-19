import unittest

import numpy as np

from strategy.metrics import closure_rate, day_frame, financial_metrics, summarize


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


class DayFrameTest(unittest.TestCase):
    """day_frame / summarize 的字段契约（webviz 与回测消费同一口径）。"""

    records = [
        {"buys": 1, "sells": 1, "grid_profit": 4.0, "grid_profit_lower": 0.0},
        {"buys": 0, "sells": 0, "grid_profit": 0.0, "grid_profit_lower": 0.0},
        {"buys": 1, "sells": 3, "grid_profit": 4.0, "grid_profit_lower": 0.0},
    ]

    def test_derived_columns(self):
        days = day_frame(self.records)

        np.testing.assert_array_equal(days["trades"], [2, 0, 4])
        np.testing.assert_array_equal(days["rounds"], [1, 0, 1])
        np.testing.assert_array_equal(days["exposure"], [0, 0, -2])

    def test_profit_per_trade_uses_each_trading_days_ratio(self):
        summary = summarize(day_frame(self.records))

        self.assertEqual(summary["n_days"], 3)
        self.assertEqual(summary["n_scored"], 2)
        # 成交日逐日收益/笔数为 2.0 与 1.0
        self.assertAlmostEqual(summary["mean_profit_per_trade"], 1.5)
        self.assertAlmostEqual(summary["std_profit_per_trade"], 0.5)

    def test_summarize_field_contract(self):
        summary = summarize(day_frame(self.records))

        for key in ("weighted_score_mean", "weighted_score_std",
                    "equal_score_mean", "equal_score_std",
                    "n_days", "n_scored", "closed_day_share",
                    "mean_rounds", "mean_buys", "mean_sells",
                    "mean_grid_profit", "std_grid_profit",
                    "mean_grid_profit_lower", "std_grid_profit_lower",
                    "mean_profit_per_trade", "std_profit_per_trade"):
            self.assertIn(key, summary)


if __name__ == "__main__":
    unittest.main()
