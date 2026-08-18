import unittest

import numpy as np

from gridscalper.metrics import financial_metrics


class FinancialMetricsTest(unittest.TestCase):
    def test_sortino_uses_zero_target_downside_deviation(self):
        metrics = financial_metrics(np.array([-0.01, -0.01]))

        self.assertAlmostEqual(metrics["SoR"], -1.0)


if __name__ == "__main__":
    unittest.main()
