# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from concurrent import futures

import grpc

import polymetis_pb2
import polymetis_pb2_grpc

from .third_party.robotiq_2finger_grippers.robotiq_2f_gripper import (
    Robotiq2FingerGripper,
)


class RobotiqGripperServer(polymetis_pb2_grpc.PolymetisControllerServerServicer):
    """gRPC server that exposes a Robotiq gripper controls to the client
    Communicates with the gripper through modbus
    """

    def __init__(self, comport):
        self.gripper = Robotiq2FingerGripper(comport=comport)

        if not self.gripper.init_success:
            raise Exception(f"Unable to open commport to {comport}")

        if not self.gripper.getStatus():
            raise Exception(f"Failed to contact gripper on port {comport}... ABORTING")

        print("Activating gripper...")
        self.gripper.activate_emergency_release()
        self.gripper.sendCommand()
        time.sleep(1)
        self.gripper.deactivate_emergency_release()
        self.gripper.sendCommand()
        time.sleep(1)
        self.gripper.activate_gripper()
        self.gripper.sendCommand()
        if (
            self.gripper.is_ready()
            and self.gripper.sendCommand()
            and self.gripper.getStatus()
        ):
            print("Activated.")
        else:
            raise Exception(f"Unable to activate!")

    def GripperGetState(self, request, context):
        self.gripper.getStatus()

        state = polymetis_pb2.GripperState()
        state.timestamp.GetCurrentTime()
        state.width = self.gripper.get_pos()
        state.max_width = self.gripper.stroke
        state.is_grasped = self.gripper.object_detected()
        state.is_moving = self.gripper.is_moving()

        return state

    def GripperGoto(self, request, context):
        self.gripper.goto(pos=request.width, vel=request.speed, force=request.force)
        self.gripper.sendCommand()

        return polymetis_pb2.Empty()

    def GripperGrasp(self, request, context):
        self.gripper.goto(pos=request.width, vel=request.speed, force=request.force)
        self.gripper.sendCommand()

        return polymetis_pb2.Empty()


class GripperServerLauncher:
    def __init__(self, ip="localhost", port="50052", comport="/dev/ttyUSB0"):
        self.address = f"{ip}:{port}"
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))

        polymetis_pb2_grpc.add_GripperServerServicer_to_server(
            RobotiqGripperServer(comport), self.server
        )
        self.server.add_insecure_port(self.address)

    def run(self):
        self.server.start()
        print(f"Robotiq-2F gripper server running at {self.address}.")
        self.server.wait_for_termination()
