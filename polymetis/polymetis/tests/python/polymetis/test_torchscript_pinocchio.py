# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

import pytest

import torch

import pybullet
import pybullet_utils.bullet_client as bc

import torchcontrol as toco
from torchcontrol.utils import test_utils

urdf_path = os.path.join(
    test_utils.project_root_dir,
    "polymetis/polymetis/data/franka_panda/panda_arm.urdf",
)

# Set default tensor type to float32
# (all tensors are float32 for all non-user code)
torch.set_default_tensor_type(torch.FloatTensor)


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


@pytest.fixture(params=["panda_link7", "panda_link8"])
def pinocchio_wrapper(request):
    link_name = request.param
    return toco.models.RobotModelPinocchio(urdf_path, link_name)


def test_incorrect_init(pinocchio_wrapper):
    with pytest.raises(RuntimeError):
        toco.models.RobotModelPinocchio(urdf_path, "panda_link_nonexistent")
    with pytest.raises(RuntimeError):
        pinocchio_wrapper.set_ee_link("panda_link_nonexistent")


@pytest.fixture
def joint_states():
    num_dofs = 7

    # Assign small random numbers to joint states
    torch.manual_seed(0)
    joint_pos = (0.1 * torch.randn(num_dofs)).tolist()
    joint_vel = (0.03 * torch.randn(num_dofs)).tolist()
    joint_acc = (0.01 * torch.randn(num_dofs)).tolist()

    return joint_pos, joint_vel, joint_acc, num_dofs


def test_link_names_and_idcs(pybullet_env, pinocchio_wrapper, joint_states):
    joint_pos, joint_vel, joint_acc, num_dofs = joint_states
    bullet_sim, robot_id = pybullet_env

    for idx in range(num_dofs):
        link_name = bullet_sim.getJointInfo(robot_id, jointIndex=idx)[12].decode(
            "utf-8"
        )
        pindex = pinocchio_wrapper.model.get_link_idx_from_name(link_name)
        pname = pinocchio_wrapper.get_link_name_from_idx(pindex)
        assert pname == link_name


def test_forward_kinematics(pybullet_env, pinocchio_wrapper, joint_states):
    joint_pos, joint_vel, joint_acc, num_dofs = joint_states
    sim, robot_id = pybullet_env
    ee_idx = get_link_idx(sim, robot_id, pinocchio_wrapper.ee_link_name)

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
    sim, robot_id = pybullet_env
    ee_idx = get_link_idx(sim, robot_id, pinocchio_wrapper.ee_link_name)

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
    sim, robot_id = pybullet_env
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

    assert torch.allclose(pinocchio_id, pyb_id, atol=1e-2)  # tolerance: 0.01 N


def test_inverse_kinematics(pybullet_env, pinocchio_wrapper, joint_states):
    max_iter = 1000
    # Setup
    joint_pos, joint_vel, joint_acc, num_dofs = joint_states
    sim, robot_id = pybullet_env
    ee_idx = get_link_idx(sim, robot_id, pinocchio_wrapper.ee_link_name)

    pinocchio_fwd_kinematics = pinocchio_wrapper.forward_kinematics(
        torch.Tensor(joint_pos)
    )
    pinocchio_pos, pinocchio_quat = pinocchio_fwd_kinematics

    # Inverse kinematics with Pinocchio
    pinocchio_joint_pos = pinocchio_wrapper.inverse_kinematics(
        pinocchio_pos, pinocchio_quat, max_iters=max_iter
    )

    ik_fwd_kinematics = pinocchio_wrapper.forward_kinematics(pinocchio_joint_pos)

    assert torch.allclose(
        pinocchio_fwd_kinematics[0], ik_fwd_kinematics[0], atol=1e-3
    ), f"Positions off: \ncurr joint pos {joint_pos}, \nIK solution {pinocchio_joint_pos}; \nactual ee pos {pinocchio_fwd_kinematics[0]}, \nik ee pos {ik_fwd_kinematics[0]}"
    assert torch.allclose(
        pinocchio_fwd_kinematics[1], ik_fwd_kinematics[1], atol=1e-3
    ), f"Positions off: \ncurr joint pos {joint_pos}, \nIK solution {pinocchio_joint_pos}; \nactual ee orient {pinocchio_fwd_kinematics[1]}, \nik ee orient {ik_fwd_kinematics[1]}"

    # Inverse kinematics with Pybullet
    pos = pinocchio_pos.tolist()
    quat = pinocchio_quat.tolist()
    pybullet_joint_pos = torch.Tensor(
        sim.calculateInverseKinematics(
            robot_id,
            ee_idx,
            pos,
            quat,
            restPoses=pinocchio_joint_pos.numpy().tolist(),
            maxNumIterations=max_iter,
        )
    )

    # Compare
    assert torch.allclose(pinocchio_joint_pos, pybullet_joint_pos, atol=1e-1)
