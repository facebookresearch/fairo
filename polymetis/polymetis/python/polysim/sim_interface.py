#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List, Callable, Optional
from dataclasses import dataclass
import io

import grpc
import torch
import hydra
from omegaconf import OmegaConf

import polymetis_pb2
import polymetis_pb2_grpc
import polymetis
from polymetis.utils import Spinner


DEFAULT_METADATA_CFG_PATH = os.path.join(
    polymetis.__path__[0], "conf/default_metadata.yaml"
)


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

    def register_arm_control(
        self,
        server_address: str,
        state_callback: Callable,
        action_callback: Callable,
        default_Kq: List[float],
        default_Kqd: List[float],
        dof: int,
        urdf_path: Optional[str] = None,
    ):
        # Construct metadata from default metadata config
        metadata_cfg = OmegaConf.load(DEFAULT_METADATA_CFG_PATH)

        metadata_cfg.hz = self.hz
        metadata_cfg.default_Kq = default_Kq
        metadata_cfg.default_Kqd = default_Kqd
        metadata_cfg.robot_model_cfg.num_dofs = dof
        if urdf_path is not None:
            metadata_cfg.robot_model_cfg.robot_description_path = urdf_path

        metadata = hydra.utils.instantiate(metadata_cfg).get_proto()

        # Connect to service
        channel = grpc.insecure_channel(server_address)
        connection = polymetis_pb2_grpc.PolymetisControllerServerStub(channel)
        connection.InitRobotClient(metadata)

        # Register callbacks
        self.control_items.append(
            ServiceInfo(connection, channel, state_callback, action_callback)
        )

    def register_gripper_control(
        self,
        server_address: str,
        state_callback: Callable,
        action_callback: Callable,
        max_width: float,
    ):
        # Construct metadata
        metadata = polymetis_pb2.GripperMetadata()
        metadata.polymetis_version = polymetis.__version__
        metadata.hz = self.hz

        metadata.max_width = max_width

        # Connect to service
        channel = grpc.insecure_channel(server_address)
        connection = polymetis_pb2_grpc.GripperServerStub(channel)
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
