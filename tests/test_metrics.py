import unittest

import numpy as np

from gridscalper.metrics import closure_rate, financial_metrics


class FinancialMetricsTest(unittest.TestCase):
    def test_sortino_uses_zero_target_downside_deviation(self):
        metrics = financial_metrics(np.array([-0.01, -0.01]))

        self.assertAlmostEqual(metrics["SoR"], -1.0)


class ClosureRateTest(unittest.TestCase):
    def test_pairing_ratio_of_one_sided_and_balanced_days(self):
        self.assertAlmostEqual(closure_rate(3, 1), 0.5)
        self.assertAlmostEqual(closure_rate(4, 4), 1.0)
        self.assertEqual(closure_rate(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
