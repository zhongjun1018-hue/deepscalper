import unittest

import torch

from gridscalper.agent import _branching_next_value


class BranchingTargetTest(unittest.TestCase):
    def test_online_selection_target_evaluation_and_inactive_gear(self):
        online = [
            torch.tensor([[1.0, 3.0], [4.0, 2.0]]),
            torch.tensor([[9.0, 1.0], [8.0, 0.0]]),
        ]
        target = [
            torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
            torch.tensor([[50.0, 60.0], [70.0, 80.0]]),
        ]

        # 样本 0 半宽选可触发档：延续价值为两支均值；样本 1 选不触发档：只取半宽分支
        value = _branching_next_value(online, target, (None, 1), inactive_gears=(0,))
        torch.testing.assert_close(value, torch.tensor([40.0, 30.0]))

        # 半宽固定为可触发档（消融）：两个样本都取两支均值
        on_value = _branching_next_value(online, target, (1, 1), inactive_gears=(0,))
        torch.testing.assert_close(on_value, torch.tensor([40.0, 60.0]))


if __name__ == "__main__":
    unittest.main()
