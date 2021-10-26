# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List
from dataclasses import dataclass
import io

import torch

import polymetis_pb2
from polymetis.utils import polymetis_version
from polymetis.utils.data_dir import get_full_path_to_urdf
from torchcontrol.policies.default_controller import DefaultController


@dataclass
class RobotModelConfig:
    """Dataclass that holds relevant information for the robot model."""

    robot_description_path: str
    controlled_joints: List[float]
    num_dofs: int
    ee_link_idx: int
    ee_link_name: str
    rest_pose: List[float]
    joint_limits_low: List[float]
    joint_limits_high: List[float]
    joint_damping: List[float]
    torque_limits: List[float]


@dataclass
class RobotClientMetadataConfig:
    """Dataclass holding full RobotClientMetadata, required for instantiatinga RobotClient with the server."""

    default_Kq: List[float]
    default_Kqd: List[float]
    hz: int
    robot_model: RobotModelConfig


class RobotClientMetadata:
    """Container class to hold all necessary metadata for the RobotClient.

    Constructs a container for the metadata by creating a default controller,
    loading the URDF file associated with the robot model and reading it into
    the metadata, and constructing the final Protobuf message containing
    the information necessary to instantiate a client that connects to the
    server.

    Args:
        default_Kq: Default position gains for the robot.

        default_Kdq: Default velocity gains for the robot.

        hz: Frequency the robot is running at.

        robot_model_cfg: A dataclass containing all the info necessary
                            for a urdf model of the robot.

    """

    def __init__(
        self,
        default_Kq: List[float],
        default_Kqd: List[float],
        hz: int,
        robot_model_cfg: RobotModelConfig,
    ):
        # Generate default controller and convert to TorchScript binary
        default_controller = DefaultController(Kq=default_Kq, Kqd=default_Kqd)
        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(default_controller), buffer)
        buffer.seek(0)
        default_controller_jitted = buffer.read()

        # Create RobotClientMetadata
        robot_client_metadata = polymetis_pb2.RobotClientMetadata()
        robot_client_metadata.hz = hz
        robot_client_metadata.dof = robot_model_cfg.num_dofs
        robot_client_metadata.ee_link_name = robot_model_cfg.ee_link_name
        robot_client_metadata.ee_link_idx = robot_model_cfg.ee_link_idx

        # Set gains as shared metadata
        robot_client_metadata.default_Kq[:] = default_Kq
        robot_client_metadata.default_Kqd[:] = default_Kqd
        robot_client_metadata.rest_pose[:] = robot_model_cfg.rest_pose

        # Set default controller for controller manager server
        robot_client_metadata.default_controller = default_controller_jitted

        # Load URDF file
        full_urdf_path = get_full_path_to_urdf(robot_model_cfg.robot_description_path)
        with open(full_urdf_path, "r") as file:
            robot_client_metadata.urdf_file = file.read()

        self.metadata_proto = robot_client_metadata

        # Set version
        robot_client_metadata.polymetis_version = polymetis_version

    def __repr__(self):
        return f"Contains protobuf message {type(self.metadata_proto)}:\n{str(self.metadata_proto)}"

    def serialize(self) -> bytes:
        """Returns a byte-serialized version of the underlying protobuf message."""
        return self.metadata_proto.SerializeToString()

    def get_proto(self):
        """Returns the underlying protobuf message."""
        return self.metadata_proto
