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
from polymetis.utils.data_dir import get_full_path_to_urdf
from torchcontrol.policies.default_controller import DefaultController


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
        return buffer.getvalue()

    def register_arm_control(
        self,
        server_address: str,
        state_callback: Callable,
        action_callback: Callable,
        dof: int,
        kp_joint: Optional[List[float]] = None,
        kd_joint: Optional[List[float]] = None,
        kp_ee: Optional[List[float]] = None,
        kd_ee: Optional[List[float]] = None,
        rest_pose: Optional[List[float]] = None,
        urdf_path: Optional[str] = None,
        ee_link_name: Optional[str] = "",
    ):
        # Construct metadata
        metadata = polymetis_pb2.RobotClientMetadata()

        metadata.polymetis_version = polymetis.__version__
        metadata.hz = self.hz
        metadata.dof = dof

        metadata.default_Kq[:] = [0.0] * dof if kp_joint is None else kp_joint
        metadata.default_Kqd[:] = [0.0] * dof if kd_joint is None else kd_joint
        metadata.default_Kx[:] = [0.0] * 6 if kp_ee is None else kp_ee
        metadata.default_Kxd[:] = [0.0] * 6 if kd_ee is None else kd_ee
        metadata.rest_pose[:] = [0.0] * dof if rest_pose is None else rest_pose

        if urdf_path is not None:
            full_urdf_path = get_full_path_to_urdf(urdf_path)
            with open(full_urdf_path, "r") as file:
                metadata.urdf_file = file.read()

        metadata.ee_link_name = ee_link_name

        default_controller = DefaultController(
            Kq=list(metadata.default_Kq), Kqd=list(metadata.default_Kqd)
        )
        metadata.default_controller = self._serialize_controller(default_controller)

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
        max_width: Optional[float] = 0.0,
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

    def run(self, time_horizon=float("inf")):
        assert self.step_callback is not None, "Step callback not assigned!"

        spinner = Spinner(self.hz)
        t = 0
        while t < time_horizon:
            # Perform control updates
            for service_info in self.control_items:
                state = service_info.state_callback()
                action = service_info.stub.ControlUpdate(state)
                service_info.action_callback(action)

            self.step_callback()

            # Spin
            t += 1
            if not self.fast_forward:
                spinner.spin()
