import time
import random
import pytest
import unittest
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
    mocked_gripper.grpc_connection.GetState.assert_called_once()
    mocked_gripper.grpc_connection.Goto.assert_called_once()
    mocked_gripper.grpc_connection.Grasp.assert_called_once()


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
    last_cmd = polymetis_pb2.GripperCommand(width=width, speed=speed, force=force)
    mocked_gripper.grpc_connection.Goto.assert_called_with(last_cmd)
