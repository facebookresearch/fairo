# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import random

import torch
import numpy as np

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from torchcontrol.modules.feedback import *
from torchcontrol.utils.test_utils import record_or_compare, FakeRobotModel

N_STATES = 4
N_ACTIONS = 3
N_DOFS = 7


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def test_linear_feedback():
    K = torch.rand(N_ACTIONS, N_STATES)
    x_current = torch.rand(N_STATES)
    x_desired = torch.rand(N_STATES)

    controller = LinearFeedback(K)

    output = controller(x_current, x_desired)

    record_or_compare("module_feedback_lf", {"output": output})


def test_joint_pd():
    Kp = torch.rand(N_DOFS, N_DOFS)
    Kd = torch.rand(N_DOFS, N_DOFS)
    joint_pos_current = torch.rand(N_DOFS)
    joint_vel_current = torch.rand(N_DOFS)
    joint_pos_desired = torch.rand(N_DOFS)
    joint_vel_desired = torch.rand(N_DOFS)

    controller = JointSpacePD(Kp=Kp, Kd=Kd)
    output = controller(
        joint_pos_current, joint_vel_current, joint_pos_desired, joint_vel_desired
    )

    record_or_compare("module_feedback_jpd", {"output": output})


def test_cartesian_pd():
    Kp = torch.rand(6, 6)
    Kd = torch.rand(6, 6)
    ee_pose_current = T.from_rot_xyz(
        translation=torch.rand(3),
        rotation=R.from_rotvec(torch.rand(3)),
    )
    ee_pose_desired = T.from_rot_xyz(
        translation=torch.rand(3),
        rotation=R.from_rotvec(torch.rand(3)),
    )
    ee_twist_current = torch.rand(6)
    ee_twist_desired = torch.rand(6)

    controller = CartesianSpacePD(Kp=Kp, Kd=Kd)
    output = controller(
        ee_pose_current, ee_twist_current, ee_pose_desired, ee_twist_desired
    )

    record_or_compare("module_feedback_cpd", {"output": output})
