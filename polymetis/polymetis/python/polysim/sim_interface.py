#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Callable, Optional
from enum import Enum
from dataclasses import dataclass
import time
import io
import json

import grpc
import numpy as np
import torch
import hydra
from omegaconf.dictconfig import DictConfig

import torchcontrol as toco
from torchcontrol.policies.default_controller import DefaultController
import polymetis_pb2
import polymetis_pb2_grpc
import polymetis
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


class SimInterface:
    def __init__(
        self,
        hz,
        fast_forward=True,
    ):
        self.hz = hz
        self.fast_forward = fast_forward

        self.control_items = []
        self.step_callback = None

    @staticmethod
    def _serialize_controller(controller):
        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(controller), buffer)
        buffer.seek(0)
        return buffer.read()

    def _construct_metadata(self, metadata_type, **kwargs):
        metadata = metadata_type()
        metadata.polymetis_version = polymetis.__version__
        metadata.hz = self.hz

        for key, item in kwargs.items():
            assert hasattr(metadata, key)
            setattr(metadata, key, item)

        return metadata

    def register_arm_control(
        self,
        server_address: str,
        state_callback: Callable,
        action_callback: Callable,
        dof: int,
        default_Kq: torch.Tensor,
        default_Kqd: torch.Tensor,
        aux_metadata: Optional[Dict] = None,
    ):
        # Load default controller
        default_Kq = torch.Tensor(default_Kq)
        default_Kqd = torch.Tensor(default_Kqd)
        assert default_Kq.shape == torch.Size([dof]) or default_Kq.shape == torch.Size(
            [dof, dof]
        )
        assert default_Kqd.shape == torch.Size(
            [dof]
        ) or default_Kqd.shape == torch.Size([dof, dof])

        default_controller = DefaultController(Kq=default_Kq, Kqd=default_Kqd)
        default_controller_jitted = self._serialize_controller(default_controller)

        # Construct metadata
        metadata = self._construct_metadata(
            polymetis_pb2.RobotClientMetadata,
            dof=dof,
            default_controller=default_controller_jitted,
            aux_metadata=json.dumps(aux_metadata or {}),
        )

        # Connect to service
        channel = grpc.insecure_channel(server_address)
        connection = polymetis_pb2_grpc.PolymetisControllerServerStub(channel)
        connection.InitRobotClient(metadata)

        # Register callbakcs
        self.control_items.append(
            ServiceInfo(connection, channel, state_callback, action_callback)
        )

    def register_gripper_control(
        self,
        server_address: str,
        state_callback: Callable,
        action_callback: Callable,
        aux_metadata: Optional[Dict] = None,
    ):
        # Construct metadata
        metadata = self._construct_metadata(
            polymetis_pb2.GripperMetadata, aux_metadata=json.dumps(aux_metadata or {})
        )

        # Connect to service
        channel = grpc.insecure_channel(server_address)
        connection = polymetis_pb2_grpc.GripperServerStub(channel)
        print(metadata)
        connection.InitRobotClient(metadata)

        # Register callbakcs
        self.control_items.append(
            ServiceInfo(connection, channel, state_callback, action_callback)
        )

    def register_step_callback(self, step_callback: Callable):
        self.step_callback = step_callback

    def run(self):
        assert self.step_callback is not None, "Step callback not assigned!"

        spinner = Spinner(self.hz)
        while True:
            # Perform control updates
            for service_info in self.control_items:
                state = service_info.state_callback()
                action = service_info.stub.ControlUpdate(state)
                service_info.action_callback(action)

            self.step_callback()

            # Spin
            if not self.fast_forward:
                spinner.spin()
