import unittest

from strategy.costs import COMMISSION_RATE, STAMP_DUTY_RATE, fee, fee_rate


class CostRateTest(unittest.TestCase):
    """显性成本口径：双边佣金 1e-4，卖出另加印花税 5e-4。"""

    def test_rate_constants(self):
        self.assertEqual(COMMISSION_RATE, 1e-4)
        self.assertEqual(STAMP_DUTY_RATE, 5e-4)

    def test_buy_pays_commission_only(self):
        self.assertEqual(fee_rate(+1), 1e-4)

    def test_sell_adds_stamp_duty(self):
        self.assertEqual(fee_rate(-1), 1e-4 + 5e-4)

    def test_fee_scales_with_notional(self):
        self.assertAlmostEqual(fee(+1, 10_000.0), 1.0)
        self.assertAlmostEqual(fee(-1, 10_000.0), 6.0)


if __name__ == "__main__":
    unittest.main()
