# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

import pytest

import torch
import numpy as np

import pybullet
import pybullet_utils.bullet_client as bc
import mujoco as mj

from torchcontrol.utils import test_utils

from omegaconf import OmegaConf
from polysim.envs import MujocoManipulatorEnv

urdf_path = os.path.join(
    test_utils.project_root_dir,
    "polymetis/polymetis/data/franka_panda/panda_arm.urdf",
)

# Set default tensor type to float32
# (all tensors are float32 for all non-user code)
torch.set_default_tensor_type(torch.FloatTensor)

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


def get_link_idx(sim, robot_id, desired_link_name):
    for i in range(sim.getNumJoints(robot_id)):
        link_name = sim.getJointInfo(robot_id, i)[12].decode("utf-8")
        if link_name == desired_link_name:
            return i
    raise Exception(f"Link {desired_link_name} not found")


@pytest.fixture
def pybullet_env():
    sim = bc.BulletClient(connection_mode=pybullet.DIRECT)
    sim.setGravity(0, 0, -9.8)
    robot_id = sim.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0],
        useFixedBase=True,
    )

    for i in range(7):
        sim.changeDynamics(
            robot_id, i, linearDamping=0, angularDamping=0, jointDamping=0
        )
    return sim, robot_id


# @pytest.fixture
def joint_states():
    num_dofs = 7

    # Assign small random numbers to joint states
    # torch.manual_seed(0)
    joint_pos = (0.1 * torch.randn(num_dofs)).tolist()
    joint_vel = (0.03 * torch.randn(num_dofs)).tolist()
    joint_acc = (0.01 * torch.randn(num_dofs)).tolist()

    return joint_pos, joint_vel, joint_acc, num_dofs


@pytest.fixture
def mj_manip_env():
    return MujocoManipulatorEnv(robot_model_cfg=franka_panda)


def test_gravity_compensation(pybullet_env, mj_manip_env):
    joint_pos, _, _, _ = joint_states()
    # vel and acc 0 for gravity compensation only
    joint_vel = [0 for _ in range(len(joint_pos))]
    joint_acc = [0 for _ in range(len(joint_pos))]

    sim, robot_id = pybullet_env
    pyb_id = torch.Tensor(
        sim.calculateInverseDynamics(robot_id, joint_pos, joint_vel, joint_acc)
    )

    set_robot_attributes(mj_manip_env, joint_pos, joint_vel, joint_acc)
    mj_id = torch.Tensor(mj_manip_env.robot_data.qfrc_bias)
    print(
        f"Inverse dynamics: Max abs diff between mj manip env & pybullet: {(pyb_id - mj_id).abs().max()}"
    )

    assert torch.allclose(mj_id, pyb_id, atol=1e-2)  # tolerance: 0.01 N


def set_robot_attributes(env, joint_pos, joint_vel, joint_acc):
    env.robot_data.qpos = joint_pos
    env.robot_data.qvel = joint_vel
    env.robot_data.qacc = joint_acc
    mj.mj_step(env.robot_model, env.robot_data)
