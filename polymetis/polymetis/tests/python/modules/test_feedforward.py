# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import random

import torch
import numpy as np

from torchcontrol.modules.feedforward import *
from torchcontrol.utils.test_utils import record_or_compare, FakeRobotModel

N_DOFS = 7


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def robot_model():
    return FakeRobotModel(N_DOFS)


@pytest.mark.parametrize("ignore_gravity", [True, False])
def test_inverse_dynamics(robot_model, ignore_gravity):
    ff_module = InverseDynamics(robot_model, ignore_gravity=ignore_gravity)

    q_current = torch.rand(N_DOFS)
    qd_current = torch.rand(N_DOFS)
    qdd_desired = torch.rand(N_DOFS)
    output = ff_module(q_current, qd_current, qdd_desired)

    record_or_compare(
        f"module_feedforward_id_{int(ignore_gravity)}", {"output": output}
    )


def test_coriolis(robot_model):
    ff_module = Coriolis(robot_model)

    q_current = torch.rand(N_DOFS)
    qd_current = torch.rand(N_DOFS)
    output = ff_module(q_current, qd_current)

    record_or_compare("module_feedforward_coriolis", {"output": output})
