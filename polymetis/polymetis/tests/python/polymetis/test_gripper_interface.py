import time
import random

import numpy as np
import pytest
from unittest.mock import MagicMock

from polymetis import GripperInterface
import polymetis_pb2


@pytest.fixture
def mocked_gripper(request):
    gripper = GripperInterface()
    gripper.grpc_connection = MagicMock()
    return gripper


@pytest.mark.parametrize("blocking", [True, False])
def test_gripper_interface(mocked_gripper, blocking):
    # Inputs
    width = 0.1
    speed = 0.2
    force = 0.3

    # Test methods
    mocked_gripper.get_state()
    mocked_gripper.goto(width=width, speed=speed, force=force, blocking=blocking)
    time.sleep(0.1)
    mocked_gripper.grasp(speed=speed, force=force, blocking=blocking)
    time.sleep(0.1)

    # Check asserts
    assert mocked_gripper.grpc_connection.GetState.call_count == 1
    assert mocked_gripper.grpc_connection.Goto.call_count == 2


@pytest.mark.parametrize("blocking", [True, False])
def test_async_gripper_commands(mocked_gripper, blocking):
    # Overload gripper with commands
    for _ in range(10):
        width = random.random()
        speed = random.random()
        force = random.random()
        mocked_gripper.goto(width=width, speed=speed, force=force, blocking=blocking)

    time.sleep(0.1)

    # Check if last goto is being executed
    last_cmd = mocked_gripper.grpc_connection.Goto.call_args_list[-1]
    np.allclose(last_cmd.args[0].width, width)
    np.allclose(last_cmd.args[0].speed, speed)
    np.allclose(last_cmd.args[0].force, force)
