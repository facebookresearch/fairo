# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from os import path
import subprocess

import torch
import pytest

from polymetis.utils.test_policies import test_parametrized_data
import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T

# Setup variables
project_root_dir = (
    subprocess.run(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    .stdout.strip()
    .decode("ascii")
)

panda_urdf_path = path.abspath(
    path.join(project_root_dir, "polymetis/polymetis/data/franka_panda/panda_arm.urdf")
)
panda_ee_link_name = "panda_link8"
robot_model = toco.models.RobotModelPinocchio(panda_urdf_path, panda_ee_link_name)

num_dofs = 7
hz = 120
time_to_go = 0.5
time_horizon = int(time_to_go * hz)

# (policy, kwargs, is_terminating, update_params)


@pytest.mark.parametrize(
    "policy_class, policy_kwargs, is_terminating, update_params", test_parametrized_data
)
def test_policy(policy_class, policy_kwargs, is_terminating, update_params):
    """
    This only tests that behavior w.r.t. time_to_go is correct.
    This test is extremely simple because position/velocity
    tracking behavior should better be done with an actual environment.
    """
    policy = policy_class(**policy_kwargs)
    scripted_policy = torch.jit.script(policy)

    with torch.no_grad():
        for t in range(time_horizon):
            assert not scripted_policy.is_terminated()
            inputs = {
                "joint_positions": torch.zeros(7),
                "joint_velocities": torch.zeros(7),
            }
            scripted_policy.forward(inputs)

        if is_terminating:
            assert scripted_policy.is_terminated()

        if update_params is not None:
            scripted_policy.update(update_params)
