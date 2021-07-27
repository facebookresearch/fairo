# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import subprocess

import pytest

import torch

import pybullet
import pybullet_utils.bullet_client as bc

import torchcontrol as toco
from torchcontrol.utils import test_utils

urdf_path = os.path.join(
    test_utils.project_root_dir,
    # "polymetis/polymetis/data/kuka_iiwa/urdf/iiwa7.urdf",
    "polymetis/polymetis/data/franka_panda/panda_arm.urdf",
)

# Set default tensor type to float32
# (all tensors are float32 for all non-user code)
torch.set_default_tensor_type(torch.FloatTensor)


@pytest.fixture
def pybullet_env():
    sim = bc.BulletClient(connection_mode=pybullet.DIRECT)
    sim.setGravity(0, 0, -9.8)
    robot_id = sim.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0],
        useFixedBase=True,
    )

    ee_idx = 7

    for i in range(7):
        sim.changeDynamics(
            robot_id, i, linearDamping=0, angularDamping=0, jointDamping=0
        )
    return sim, robot_id, ee_idx


@pytest.fixture
def pinocchio_wrapper():
    return toco.models.RobotModelPinocchio(urdf_path, "panda_link8")


@pytest.fixture
def joint_states():
    num_dofs = 7
    joint_pos = [0 for _ in range(num_dofs)]
    joint_vel = [0 for _ in range(num_dofs)]
    joint_acc = [0 for _ in range(num_dofs)]
    return joint_pos, joint_vel, joint_acc, num_dofs


def test_forward_kinematics(pybullet_env, pinocchio_wrapper, joint_states):
    joint_pos, joint_vel, joint_acc, num_dofs = joint_states
    sim, robot_id, ee_idx = pybullet_env

    pinocchio_fwd_kinematics = pinocchio_wrapper.forward_kinematics(
        torch.Tensor(joint_pos)
    )
    pinocchio_pos, pinocchio_quat = pinocchio_fwd_kinematics

    for i in range(num_dofs):
        sim.resetJointState(
            bodyUniqueId=robot_id,
            jointIndex=i,
            targetValue=joint_pos[i],
            targetVelocity=joint_vel[i],
        )
    pybullet_fwd_kinematics = sim.getLinkState(
        robot_id, ee_idx, computeForwardKinematics=True
    )
    pybullet_pos = torch.Tensor(pybullet_fwd_kinematics[4])
    pybullet_quat = torch.Tensor(pybullet_fwd_kinematics[5])

    assert torch.allclose(pinocchio_pos, pybullet_pos)
    assert torch.allclose(pinocchio_quat, pybullet_quat)


def test_jacobians(pybullet_env, pinocchio_wrapper, joint_states):
    joint_pos, joint_vel, joint_acc, num_dofs = joint_states
    sim, robot_id, ee_idx = pybullet_env

    pybullet_jacobian = torch.Tensor(
        sim.calculateJacobian(
            robot_id, ee_idx, [0, 0, 0], joint_pos, joint_vel, joint_acc
        )
    )
    pinocchio_jacobian = pinocchio_wrapper.compute_jacobian(
        torch.Tensor(joint_pos)
    ).reshape(pybullet_jacobian.shape)

    print(
        f"Jacobian: Max abs diff between pinocchio & pybullet: {(pybullet_jacobian - pinocchio_jacobian).abs().max()}"
    )
    assert torch.allclose(pybullet_jacobian, pinocchio_jacobian)


def test_inverse_dynamics(pybullet_env, pinocchio_wrapper, joint_states):
    joint_pos, joint_vel, joint_acc, num_dofs = joint_states
    sim, robot_id, ee_idx = pybullet_env
    pinocchio_id = pinocchio_wrapper.inverse_dynamics(
        torch.Tensor(joint_pos).unsqueeze(1),
        torch.Tensor(joint_vel).unsqueeze(1),
        torch.Tensor(joint_acc).unsqueeze(1),
    )
    pyb_id = torch.Tensor(
        sim.calculateInverseDynamics(robot_id, joint_pos, joint_vel, joint_acc)
    )
    print(
        f"Inverse dynamics: Max abs diff between pinocchio & pybullet: {(pyb_id - pinocchio_id).abs().max()}"
    )

    assert torch.allclose(pinocchio_id, pyb_id, atol=1e-4)


def test_inverse_kinematics(pybullet_env, pinocchio_wrapper, joint_states):
    # Setup
    joint_pos, joint_vel, joint_acc, num_dofs = joint_states
    sim, robot_id, ee_idx = pybullet_env

    pinocchio_fwd_kinematics = pinocchio_wrapper.forward_kinematics(
        torch.Tensor(joint_pos)
    )
    pinocchio_pos, pinocchio_quat = pinocchio_fwd_kinematics

    # Inverse kinematics with Pinocchio
    pinocchio_joint_pos = pinocchio_wrapper.inverse_kinematics(
        pinocchio_pos, pinocchio_quat, max_iters=1
    )

    # Inverse kinematics with Pybullet
    pos = pinocchio_pos.tolist()
    quat = pinocchio_quat.tolist()
    pybullet_joint_pos = torch.Tensor(
        sim.calculateInverseKinematics(robot_id, ee_idx, pos, quat)
    )

    # Compare
    assert torch.allclose(pinocchio_joint_pos, pybullet_joint_pos)
