import unittest

import torch

from control.agent import _branching_next_value


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
        allowed = torch.tensor([True, True])

        # 各分支由 online 网络选档、target 网络估值。样本 0 半宽选可触发档：延续价值
        # 为两支均值 (20+50)/2；样本 1 选不触发档：只取半宽分支 30
        value = _branching_next_value(online, target, inactive_gears=(0,),
                                      flatten_allowed=allowed)
        torch.testing.assert_close(value, torch.tensor([35.0, 30.0]))

    def test_flatten_mask_excludes_gear_zero_from_argmax(self):
        online = [
            torch.tensor([[1.0, 3.0], [4.0, 2.0]]),
            torch.tensor([[9.0, 1.0], [8.0, 0.0]]),
        ]
        target = [
            torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
            torch.tensor([[50.0, 60.0], [70.0, 80.0]]),
        ]

        # 样本 1 净持仓为零：平仓档（半宽档 0）被掩码，argmax 落到档 1，
        # 该档可触发，延续价值取两支均值 (40+70)/2
        value = _branching_next_value(online, target, inactive_gears=(0,),
                                      flatten_allowed=torch.tensor([True, False]))
        torch.testing.assert_close(value, torch.tensor([35.0, 55.0]))


if __name__ == "__main__":
    unittest.main()
