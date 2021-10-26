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
    mocked_gripper.grasp(speed=speed, force=force, blocking=blocking)

    # Check asserts
    mocked_gripper.grpc_connection.GetState.assert_called_once()
    mocked_gripper.grpc_connection.Goto.assert_called_once()
    mocked_gripper.grpc_connection.Grasp.assert_called_once()
