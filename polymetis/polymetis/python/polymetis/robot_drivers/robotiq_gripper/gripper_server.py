# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from concurrent import futures

import grpc

import polymetis_pb2
import polymetis_pb2_grpc

from robotiq_2f_gripper_control.robotiq_2f_gripper import Robotiq2FingerGripper


class RobotiqGripperServer(polymetis_pb2_grpc.GripperServerServicer):
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

    def GetState(self, request, context):
        state = polymetis_pb2.GripperState()

        state.timestamp.GetCurrentTime()
        state.pos = self.gripper.get_pos()
        state.is_ready = self.gripper.is_ready()
        state.is_moving = self.gripper.is_moving()
        state.is_stopped = self.gripper.is_stopped()

        return state

    def Goto(self, request, context):
        self.gripper.goto(pos=request.pos, vel=request.vel, force=request.force)
        self.gripper.sendCommand()

        return polymetis_pb2.Empty()


def run_server(ip, port, comport):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))

    polymetis_pb2_grpc.add_GripperServerServicer_to_server(
        RobotiqGripperServer(comport), server
    )
    server.add_insecure_port(f"{ip}:{port}")
    server.start()
    server.wait_for_termination()
