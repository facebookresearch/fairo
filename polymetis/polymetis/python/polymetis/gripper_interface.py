# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import threading

import grpc

import polymetis_pb2
import polymetis_pb2_grpc


EMPTY = polymetis_pb2.Empty()


class GripperInterface:
    """Gripper interface class to initialize a connection to a gRPC gripper server.

    Args:
        ip_address: IP address of the gRPC-based gripper server.
        port: Port to connect to on the IP address.
    """

    def __init__(self, ip_address: str = "localhost", port: int = 50052):
        self.channel = grpc.insecure_channel(f"{ip_address}:{port}")
        self.grpc_connection = polymetis_pb2_grpc.GripperServerStub(self.channel)

    def _send_gripper_command(self, command, msg, blocking=True) -> None:
        if blocking:
            command(msg)
        else:
            threading.Thread(target=command, args=(msg,)).start()

    def get_state(self) -> polymetis_pb2.GripperState:
        """Returns the state of the gripper
        Returns:
            gripper state (polymetis_pb2.GripperState)
        """
        return self.grpc_connection.GetState(EMPTY)

    def goto(self, width, speed, force):
        """Commands the gripper to a certain width
        Args:
            pos: Target width
            vel: Velocity of the movement
            force: Maximum force the gripper will exert
        """
        self.grpc_connection.Goto(
            polymetis_pb2.GripperCommand(width=width, speed=speed, force=force)
        )

    def grasp(self, speed, force):
        """Commands the gripper to a certain width
        Args:
            vel: Velocity of the movement
            force: Maximum force the gripper will exert
        """
        self.grpc_connection.Grasp(
            polymetis_pb2.GripperCommand(width=0.0, speed=speed, force=force)
        )
