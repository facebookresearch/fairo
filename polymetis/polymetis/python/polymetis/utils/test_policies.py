# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from polymetis.utils.data_dir import get_full_path_to_urdf

import torchcontrol as toco
from torchcontrol.transform import Transformation as T
from torchcontrol.transform import Rotation as R


panda_urdf_path = get_full_path_to_urdf("franka_panda/panda_arm.urdf")
panda_ee_link_name = "panda_link8"
robot_model = toco.models.RobotModelPinocchio(panda_urdf_path, panda_ee_link_name)

num_dofs = 7
hz = 120
time_to_go = 0.5
time_horizon = int(time_to_go * hz)

test_parametrized_data = [
    (
        toco.policies.JointImpedanceControl,
        dict(
            joint_pos_current=torch.rand(num_dofs),
            Kp=torch.rand(num_dofs, num_dofs),
            Kd=torch.rand(num_dofs, num_dofs),
            robot_model=robot_model,
        ),
        False,
        {"joint_pos_desired": torch.rand(num_dofs)},
    ),
    (
        toco.policies.CartesianImpedanceControl,
        dict(
            joint_pos_current=torch.rand(num_dofs),
            Kp=torch.rand(6, 6),
            Kd=torch.rand(6, 6),
            robot_model=robot_model,
        ),
        False,
        {"ee_pos_desired": torch.rand(3)},
    ),
    (
        toco.policies.HybridJointImpedanceControl,
        dict(
            joint_pos_current=torch.rand(num_dofs),
            Kq=torch.rand(num_dofs, num_dofs),
            Kqd=torch.rand(num_dofs, num_dofs),
            Kx=torch.rand(6, 6),
            Kxd=torch.rand(6, 6),
            robot_model=robot_model,
        ),
        False,
        {"joint_pos_desired": torch.rand(num_dofs)},
    ),
    (
        toco.policies.JointTrajectoryExecutor,
        dict(
            joint_pos_trajectory=[torch.rand(num_dofs) for _ in range(time_horizon)],
            joint_vel_trajectory=[torch.rand(num_dofs) for _ in range(time_horizon)],
            Kq=torch.rand(num_dofs, num_dofs),
            Kqd=torch.rand(num_dofs, num_dofs),
            Kx=torch.rand(6, 6),
            Kxd=torch.rand(6, 6),
            robot_model=robot_model,
            ignore_gravity=True,
        ),
        True,
        None,
    ),
    (
        toco.policies.EndEffectorTrajectoryExecutor,
        dict(
            ee_pose_trajectory=[
                T.from_rot_xyz(
                    rotation=R.from_rotvec(torch.rand(3)), translation=torch.rand(3)
                )
                for _ in range(time_horizon)
            ],
            ee_twist_trajectory=[torch.rand(6) for _ in range(time_horizon)],
            Kp=torch.rand(6, 6),
            Kd=torch.rand(6, 6),
            robot_model=robot_model,
            ignore_gravity=True,
        ),
        True,
        None,
    ),
    (
        toco.policies.iLQR,
        dict(
            Kxs=torch.rand(time_horizon, num_dofs, 2 * num_dofs),
            x_desireds=torch.rand(time_horizon, 2 * num_dofs),
            u_ffs=torch.rand(time_horizon, num_dofs),
        ),
        True,
        None,
    ),
]
