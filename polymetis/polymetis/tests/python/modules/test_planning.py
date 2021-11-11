# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import random

import torch

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T
from torchcontrol.utils.test_utils import record_or_compare, FakeRobotModel


N_DOFS = 7
TIME_TO_GO = 3


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(0)
    torch.manual_seed(0)


@pytest.fixture(params=[2, 25])
def num_steps(request):
    return request.param


def test_joint_planner(num_steps):
    joint_start = torch.rand(N_DOFS)
    joint_goal = torch.rand(N_DOFS)

    waypoints = toco.planning.generate_joint_space_min_jerk(
        start=joint_start,
        goal=joint_goal,
        time_to_go=TIME_TO_GO,
        hz=num_steps / TIME_TO_GO,
    )

    q_ls = [waypoint["position"] for waypoint in waypoints]
    qd_ls = [waypoint["velocity"] for waypoint in waypoints]
    qdd_ls = [waypoint["acceleration"] for waypoint in waypoints]

    assert torch.allclose(q_ls[0], joint_start)
    assert torch.allclose(q_ls[-1], joint_goal)

    output_dict = {
        "q_arr": torch.stack(q_ls),
        "qd_arr": torch.stack(qd_ls),
        "qdd_arr": torch.stack(qdd_ls),
    }
    record_or_compare(f"module_planning_joints_{num_steps}", output_dict)


def test_cartesian_planner(num_steps):
    pose_start = T.from_rot_xyz(
        translation=torch.rand(3),
        rotation=R.from_rotvec(torch.rand(3)),
    )
    pose_goal = T.from_rot_xyz(
        translation=torch.rand(3),
        rotation=R.from_rotvec(torch.rand(3)),
    )

    waypoints = toco.planning.generate_cartesian_space_min_jerk(
        start=pose_start,
        goal=pose_goal,
        time_to_go=TIME_TO_GO,
        hz=num_steps / TIME_TO_GO,
    )

    qx_ls = [waypoint["pose"].translation() for waypoint in waypoints]
    qr_ls = [waypoint["pose"].rotation().as_quat() for waypoint in waypoints]
    qd_ls = [waypoint["twist"] for waypoint in waypoints]
    qdd_ls = [waypoint["acceleration"] for waypoint in waypoints]

    assert torch.allclose(qx_ls[0], pose_start.translation())
    assert torch.allclose(qr_ls[0], pose_start.rotation().as_quat())
    assert torch.allclose(qx_ls[-1], pose_goal.translation())
    assert torch.allclose(qr_ls[-1], pose_goal.rotation().as_quat())

    output_dict = {
        "qx_arr": torch.stack(qx_ls),
        "qr_arr": torch.stack(qr_ls),
        "qd_arr": torch.stack(qd_ls),
        "qdd_arr": torch.stack(qdd_ls),
    }
    record_or_compare(f"module_planning_cartesian_{num_steps}", output_dict)


def test_cartesian_position_planner(num_steps):
    pos_start = torch.rand(3)
    pos_goal = torch.rand(3)

    waypoints = toco.planning.generate_position_min_jerk(
        start=pos_start, goal=pos_goal, time_to_go=TIME_TO_GO, hz=num_steps / TIME_TO_GO
    )

    q_ls = [waypoint["position"] for waypoint in waypoints]
    qd_ls = [waypoint["velocity"] for waypoint in waypoints]
    qdd_ls = [waypoint["acceleration"] for waypoint in waypoints]

    assert torch.allclose(q_ls[0], pos_start)
    assert torch.allclose(q_ls[-1], pos_goal)

    output_dict = {
        "q_arr": torch.stack(q_ls),
        "qd_arr": torch.stack(qd_ls),
        "qdd_arr": torch.stack(qdd_ls),
    }
    record_or_compare(f"module_planning_pos_{num_steps}", output_dict)


def test_cartesian_joint_planner(num_steps):
    joint_start = torch.rand(N_DOFS)
    pose_goal = T.from_rot_xyz(
        translation=torch.rand(3),
        rotation=R.from_quat(torch.Tensor([0, 0, 0, 1])),
    )
    robot_model = FakeRobotModel(N_DOFS)

    waypoints = toco.planning.generate_cartesian_target_joint_min_jerk(
        joint_pos_start=joint_start,
        ee_pose_goal=pose_goal,
        time_to_go=TIME_TO_GO,
        hz=num_steps / TIME_TO_GO,
        robot_model=robot_model,
    )

    q_ls = [waypoint["position"] for waypoint in waypoints]
    qd_ls = [waypoint["velocity"] for waypoint in waypoints]
    qdd_ls = [waypoint["acceleration"] for waypoint in waypoints]

    assert torch.allclose(q_ls[0], joint_start)

    output_dict = {
        "q_arr": torch.stack(q_ls),
        "qd_arr": torch.stack(qd_ls),
        "qdd_arr": torch.stack(qdd_ls),
    }
    record_or_compare(f"module_planning_cartesian_joints_{num_steps}", output_dict)
