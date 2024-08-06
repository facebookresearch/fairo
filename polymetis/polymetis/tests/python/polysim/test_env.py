# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest

import os
import numpy as np

from omegaconf import OmegaConf
from polysim.envs import BulletManipulatorEnv
from polysim.envs import HabitatManipulatorEnv
from polysim.envs import MujocoManipulatorEnv

import pybullet_data

from polymetis_pb2 import RobotState

# from polysim.envs import DaisyLocomotorEnv

franka_panda = OmegaConf.create(
    {
        "robot_description_path": "franka_panda/panda_arm.urdf",
        "controlled_joints": [0, 1, 2, 3, 4, 5, 6],
        "ee_link_idx": 7,
        "ee_link_name": "panda_link8",
        "rest_pose": [
            -0.13935425877571106,
            -0.020481698215007782,
            -0.05201413854956627,
            -2.0691256523132324,
            0.05058913677930832,
            2.0028650760650635,
            -0.9167874455451965,
        ],
        "joint_limits_low": [
            -2.8973,
            -1.7628,
            -2.8973,
            -3.0718,
            -2.8973,
            -0.0175,
            -2.8973,
        ],
        "joint_limits_high": [
            2.8973,
            1.7628,
            2.8973,
            -0.0698,
            2.8973,
            3.7525,
            2.8973,
        ],
        "joint_damping": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "torque_limits": [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
        "use_grav_comp": True,
        "num_dofs": 7,
    },
)

kuka_iiwa = OmegaConf.create(
    {
        "robot_description_path": "kuka_iiwa/urdf/iiwa7.urdf",
        "controlled_joints": [0, 1, 2, 3, 4, 5, 6],
        "ee_link_idx": 7,
        "ee_link_name": "panda_link8",
        "rest_pose": [0.0, 0.0, 0.0, -1.1, 0.0, 1.0, 0.0],
        "joint_limits_low": [
            -2.9671,
            -2.0944,
            -2.9671,
            -2.0944,
            -2.9671,
            -2.0944,
            -3.0543,
        ],
        "joint_limits_high": [
            2.9671,
            2.0944,
            2.9671,
            2.0944,
            2.9671,
            2.0944,
            3.0543,
        ],
        "joint_damping": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "torque_limits": [30.0, 30.0, 30.0, 30.0, 30.0, 20.0, 20.0],
        "use_grav_comp": True,
        "num_dofs": 7,
    },
)

kuka_gripper_sdf = OmegaConf.create(
    {
        "num_dofs": 12,
        "robot_state_dim": 12,
        "hz": 240,
        "time_warp": 1.0,
        "default_controller_args": {
            "type": "default",
            "Kx": [
                100.0,
                150.0,
                50.0,
                75.0,
                25.0,
                10.0,
                10.0,
                10.0,
                20.0,
                5.0,
                10.0,
                2.5,
                # 1.0,
                # 0.5,
            ],
            "pos_state_dim": 7,
        },
        "robot_description_path": os.path.join(
            pybullet_data.getDataPath(), "kuka_iiwa/kuka_with_gripper2.sdf"
        ),
        "controlled_joints": [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 10, 13],
        "rest_pose": [
            0.006418,
            0.413184,
            -0.011401,
            -1.589317,
            0.005379,
            1.137684,
            -0.006539,
            0.000048,
            -0.299912,
            0.000000,
            -0.000043,
            0.299960,
            # 0.000000,
            # -0.000200,
        ],
        "joint_limits_high": [
            2.96705972839,
            2.09439510239,
            2.96705972839,
            2.09439510239,
            2.96705972839,
            2.09439510239,
            3.05432619099,
            2.09439510239,
            2.96705972839,
            # 2.09439510239,
            # 3.05432619099,
        ],
        "joint_limits_low": [
            -2.96705972839,
            -2.09439510239,
            -2.96705972839,
            -2.09439510239,
            -2.96705972839,
            -2.09439510239,
            -3.05432619099,
            -2.09439510239,
            -2.96705972839,
            # -2.09439510239,
            # -3.05432619099,
        ],
        "joint_damping": None,
        "torque_limits": [200, 200, 100, 100, 100, 30, 30, 10, 10, 10, 10, 10],
        "ee_link_idx": 5,
        "ee_link_name": "J5",
        "using_camera": False,
        "gpu_renderer": False,
    },
)


@pytest.mark.parametrize(
    "obj, obj_kwargs",
    [
        (
            BulletManipulatorEnv,
            {
                "robot_model_cfg": franka_panda,
            },
        ),
        (
            BulletManipulatorEnv,
            {
                "robot_model_cfg": kuka_iiwa,
            },
        ),
        (
            BulletManipulatorEnv,
            {
                "robot_model_cfg": kuka_gripper_sdf,
            },
        ),
        (
            HabitatManipulatorEnv,
            {
                "robot_model_cfg": franka_panda,
                "habitat_dir": os.path.join(
                    os.path.dirname(__file__), "../../data/habitat-sim"
                ),
            },
        ),
        (
            MujocoManipulatorEnv,
            {
                "robot_model_cfg": franka_panda,
            },
        ),
    ],
)
def test_env(obj, obj_kwargs):
    # Initialize env
    env = obj(**obj_kwargs, gui=False)

    # Test env functionalities
    env.reset()
    env.get_current_joint_pos_vel()
    env.get_current_joint_torques()
    env.apply_joint_torques(np.zeros(env.get_num_dofs()))


@pytest.mark.parametrize(
    "obj, obj_kwargs",
    [
        (
            BulletManipulatorEnv,
            {
                "robot_model_cfg": franka_panda,
            },
        ),
        (
            BulletManipulatorEnv,
            {
                "robot_model_cfg": kuka_iiwa,
            },
        ),
        # (
        #     MujocoManipulatorEnv,
        #     {
        #         "robot_model_cfg": franka_panda,
        #     },
        # ),
    ],
)
def test_mirror_env(obj, obj_kwargs):
    env = obj(**obj_kwargs, gui=False)
    ndofs = obj_kwargs["robot_model_cfg"].num_dofs
    robot_state = RobotState()
    robot_state.joint_positions[:] = list(obj_kwargs["robot_model_cfg"].rest_pose)
    robot_state.joint_velocities[:] = np.zeros(ndofs)
    env.set_robot_state(robot_state)
    obs_pos, obs_vel = env.get_current_joint_pos_vel()
    assert np.allclose(robot_state.joint_positions, obs_pos, atol=1e-3)
    assert np.allclose(robot_state.joint_velocities, obs_vel, atol=1e-3)
