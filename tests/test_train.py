import unittest

from gridscalper.train import eval_days


class EvalDaysTest(unittest.TestCase):
    def test_checkpoints_split_days_evenly_and_end_at_last_day(self):
        self.assertEqual(eval_days(68, 3), {21, 44, 67})
        self.assertEqual(eval_days(9, 1), {8})
        self.assertEqual(eval_days(2, 3), {0, 1})   # 评估点多于交易日时逐日评估


if __name__ == "__main__":
    unittest.main()
