import unittest

from data_provider.split import SPLIT_RATIOS, chronological_split


class ChronologicalSplitTest(unittest.TestCase):
    def test_default_ratios_are_7_1_2(self):
        self.assertEqual(SPLIT_RATIOS, (0.7, 0.1, 0.2))

    def test_segments_follow_the_ratios(self):
        dates = [f"202601{day:02d}" for day in range(1, 11)]
        split = chronological_split(dates)

        self.assertEqual((len(split.train), len(split.val), len(split.test)), (7, 1, 2))

    def test_segments_are_ordered_and_non_overlapping(self):
        dates = ["20260103", "20260101", "20260105", "20260102", "20260104",
                 "20260106", "20260107", "20260108", "20260109", "20260110"]
        split = chronological_split(dates)

        self.assertEqual(split.train + split.val + split.test, sorted(dates))
        self.assertLess(max(split.train), min(split.val))
        self.assertLess(max(split.val), min(split.test))

    def test_empty_segment_is_rejected(self):
        with self.assertRaises(ValueError):
            chronological_split(["20260101", "20260102", "20260103"])  # 验证段为空


if __name__ == "__main__":
    unittest.main()
