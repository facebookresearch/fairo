# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import threading
import queue

import grpc

import polymetis_pb2
import polymetis_pb2_grpc


log = logging.getLogger(__name__)

EMPTY = polymetis_pb2.Empty()


class GripperInterface:
    """Gripper interface class to initialize a connection to a gRPC gripper server.

    Args:
        ip_address: IP address of the gRPC-based gripper server.
        port: Port to connect to on the IP address.
    """

    def __init__(self, ip_address: str = "localhost", port: int = 50052):
        # Connect to server
        self.channel = grpc.insecure_channel(f"{ip_address}:{port}")
        self.grpc_connection = polymetis_pb2_grpc.GripperServerStub(self.channel)

        # Get metadata
        try:
            self.metadata = self.grpc_connection.GetRobotClientMetadata(EMPTY)
        except grpc.RpcError:
            log.warning("Metadata unavailable from server.")

        # Execute commands from cache in separate thread
        self._command_thr = threading.Thread(
            target=self._command_executor,
            daemon=True,
        )

        self._command_queue = queue.Queue(maxsize=1)
        self._command_thr.start()

    def _command_executor(self):
        while True:
            command, msg = self._command_queue.get()
            try:
                command(msg)
            except grpc.RpcError as e:
                raise grpc.RpcError(f"GRIPPER SERVER ERROR --\n{e.details()}") from None
            self._command_queue.task_done()

    def _send_gripper_command(self, command, msg, blocking: bool = True) -> None:
        self._command_queue.put((command, msg))

        if blocking:
            self._command_queue.join()

    def get_state(self) -> polymetis_pb2.GripperState:
        """Returns the state of the gripper
        Returns:
            gripper state (polymetis_pb2.GripperState)
        """
        return self.grpc_connection.GetState(EMPTY)

    def goto(self, width: float, speed: float, force: float, blocking: bool = True):
        """Commands the gripper to a certain width
        Args:
            pos: Target width
            vel: Velocity of the movement
            force: Maximum force the gripper will exert
        """
        cmd = polymetis_pb2.GripperCommand(
            width=width, speed=speed, force=force, grasp=False
        )
        cmd.timestamp.GetCurrentTime()

        self._send_gripper_command(
            self.grpc_connection.Goto,
            cmd,
            blocking=blocking,
        )

    def grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        """Commands the gripper to a close

        For Robotiq grippers, this is equivalent to calling `goto` with grasp_width (0 by default).

        For the Franka Hand, see documentation for franka::Gripper::move in libfranka.
        Basically the gripper will perform the grasp if grasp_width - epsilon_inner < final_width < grasp_width + epsilon_outer.
        Else the gripper will not continue to exert force.
        The default width and epsilon values ensures that the gripper closes to the minimum width possible and continues to exert force.


        Args:
            vel: Velocity of the movement
            force: Maximum force the gripper will exert
            grasp_width: Target width of the grasp
            epsilon_inner: Maximum tolerated deviation when the actual grasped width is smaller than the commanded grasp width
            epsilon_outer: Maximum tolerated deviation when the actual grasped width is larger than the commanded grasp width
        """
        cmd = polymetis_pb2.GripperCommand(
            width=grasp_width,
            speed=speed,
            force=force,
            grasp=True,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
        )
        cmd.timestamp.GetCurrentTime()

        self._send_gripper_command(
            self.grpc_connection.Goto,
            cmd,
            blocking=blocking,
        )
