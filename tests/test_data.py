import unittest

from gridscalper.data import walk_forward_splits


class WalkForwardSplitTest(unittest.TestCase):
    days = list(range(119))

    def test_single_fold_is_the_ratio_split(self):
        (train, val, test), = walk_forward_splits(self.days, 0.65, 0.15, 1)

        self.assertEqual((len(train), len(val), len(test)), (77, 17, 25))
        self.assertEqual(test[-1], self.days[-1])

    def test_earlier_fold_shifts_back_by_one_test_window(self):
        folds = walk_forward_splits(self.days, 0.65, 0.15, 2)

        self.assertEqual(folds[-1], walk_forward_splits(self.days, 0.65, 0.15, 1)[0])
        self.assertEqual(folds[0], (self.days[:52], self.days[52:69], self.days[69:94]))

    def test_folds_beyond_available_history_are_rejected(self):
        with self.assertRaises(ValueError):
            walk_forward_splits(self.days, 0.65, 0.15, 5)


if __name__ == "__main__":
    unittest.main()
