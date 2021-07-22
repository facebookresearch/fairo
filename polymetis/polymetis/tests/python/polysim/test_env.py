# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest

import os
import numpy as np

from omegaconf import DictConfig
from polysim.envs import BulletManipulatorEnv
from polysim.envs import HabitatManipulatorEnv

# from polysim.envs import DaisyLocomotorEnv

franka_panda = DictConfig(
    {
        "robot_description_path": "franka_panda/panda_arm.urdf",
        "controlled_joints": [0, 1, 2, 3, 4, 5, 6],
        "ee_link_idx": 7,
        "ee_joint_name": "panda_joint8",
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
    }
)

kuka_iiwa = DictConfig(
    {
        "robot_description_path": "kuka_iiwa/urdf/iiwa7.urdf",
        "controlled_joints": [0, 1, 2, 3, 4, 5, 6],
        "ee_link_idx": 7,
        "ee_joint_name": "panda_joint8",
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
    }
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
            HabitatManipulatorEnv,
            {
                "robot_model_cfg": franka_panda,
                "habitat_dir": os.path.join(
                    os.path.dirname(__file__), "../../data/habitat-sim"
                ),
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
