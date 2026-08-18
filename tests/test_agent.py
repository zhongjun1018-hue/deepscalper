import unittest

import torch

from gridscalper.agent import _branching_next_value


class BranchingTargetTest(unittest.TestCase):
    def test_online_selection_target_evaluation_and_fixed_gear(self):
        online = [
            torch.tensor([[1.0, 3.0], [4.0, 2.0]]),
            torch.tensor([[9.0, 1.0], [8.0, 0.0]]),
        ]
        target = [
            torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
            torch.tensor([[50.0, 60.0], [70.0, 80.0]]),
        ]

        value = _branching_next_value(online, target, (None, 1), off_gear=0)

        torch.testing.assert_close(value, torch.tensor([40.0, 55.0]))

        off_value = _branching_next_value(online, target, (None, 0), off_gear=0)
        torch.testing.assert_close(off_value, torch.tensor([50.0, 70.0]))


if __name__ == "__main__":
    unittest.main()
