#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable
import enum
import time
import numpy as np
import hydra
from omegaconf.dictconfig import DictConfig

from polymetis.utils import Spinner
from polymetis.robot_client.abstract_robot_client import (
    AbstractRobotClient,
)


class ControlType(Enum):
    ARM = 0
    GRIPPER = 1


@dataclass
class ServiceInfo:
    stub: object
    channel: object
    state_callback: Callable
    action_callback: Callable


class SimInterface(AbstractRobotClient):
    def __init__(
        self,
        hz,
        intraprocess=False,  # TODO
    ):
        self.hz = hz

        self.control_items = []
        self.step_callback = None

    def register_control_callback(
        self,
        server_ip: str,
        server_port: str,
        server_type: ControlType,
        state_callback: Callable,
        action_callback: Callable,
    ):
        # Connect to server
        channel = grpc.insecure_channel(f"{ip}:{port}")
        if server_type is ControlType.ARM:
            connection = polymetis_pb2_grpc.PolymetisControllerServerStub(channel)
        elif server_type is ControlType.ARM:
            connection = polymetis_pb2_grpc.GripperServerStub(channel)
        else:
            raise AttributeError("Invalid server type.")

        # Register control item
        self.control_items.append(
            ServiceInfo(connection, channel, state_callback, action_callback)
        )

    def register_step_callback(self, step_callback: Callable):
        self.step_callback = step_callback

    def run(self):
        assert self.step_callback is not None, "Step callback not assigned!"

        spinner = Spinner(self.metadata.get_proto().hz)
        while True:
            # Perform control updates
            for service_info in self.control_items:
                state = service_info.state_callback()
                action = service_info.stub.ControlUpdate(state)
                service_info.action_callback(action)

            self.step_callback()

            # Spin
            spinner.spin()
