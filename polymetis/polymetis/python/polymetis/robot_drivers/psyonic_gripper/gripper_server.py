import sys
import os
import time
import io
from concurrent import futures

import grpc
import hydra
import torch

import polymetis
import polymetis_pb2
import polymetis_pb2_grpc

import numpy as np

d = os.path.dirname(__file__)
sys.path.append(
    os.path.join(d, "ability-hand-api", "python", "psyonic-ability-hand", "src")
)

from psyonic_ability_hand.hand import Hand, MockComm, JointData
from psyonic_ability_hand.io import MockIO, SerialIO


class PsyonicGripperServer(polymetis_pb2_grpc.PolymetisControllerServerServicer):
    """gRPC server that exposes Psyonic Ability Hand controls to the client Communicates
    with the gripper through RS-485. We are using the PolymetisControllerServer
    interface, rather than the GripperServer interface, because it allows us to expose
    more of the functionality of the hand.
    """

    def __init__(self, hand, metadata):
        self._hand = hand
        self._metadata = metadata

    def SetController(self, request, context):
        # determine frequency to execute commands
        hz = self._metadata.get_proto().hz

        # combine chunks into complete bytes string
        policy = b""
        for chunk in request:
            policy += chunk.torchscript_binary_chunk

        # convert bytes string to torch policy and extract waypoints
        buffer = io.BytesIO(policy)
        policy = torch.jit.load(buffer)
        waypoints = policy.joint_pos_trajectory.tolist()

        # iterate through waypoints and set hand to joint angles
        for angle_set in waypoints:
            joint_angles = JointData(
                *angle_set[:-4]
            )  # declare JointData but remove 4 zeros from each angle set.
            self._hand.set_position(joint_angles)
            time.sleep(1 / hz)

        # self._hand.set_velocity(joint_angles)
        # self._hand.set_torque(joint_angles)

        return polymetis_pb2.LogInterval()

    def UpdateController(self, request, context):
        return polymetis_pb2.LogInterval()

    def TerminateController(self, request, context):
        return polymetis_pb2.LogInterval()

    def GetRobotState(self, request, context):
        """Returns the current state of the robot"""
        print("ROBOT STATE REQUEST", request)

        position = np.array(self._hand.position.to_list())  # length 6
        velocity = np.array(self._hand.velocity.to_list())  # length 6
        current = np.array(self._hand.current.to_list())  # length 6
        touch = np.array([np.array(f) for f in self._hand.touch.to_list()])

        print(f"position: {position}")
        print(f"current: {current}")
        print(f"touch: {touch}")

        # TODO: populate the fields of the robot state message with data from the hand.
        # Joint torques need to be computed from current values.
        state = polymetis_pb2.RobotState(
            joint_positions=np.concatenate(
                (position, [0, 0, 0, 0])
            ),  # URDF shows 10 revolute joints, but hand only offers control of 6 of those? So we add 4 extra zeros to the position array.
            joint_velocities=np.concatenate((velocity, [0, 0, 0, 0])),
        )

        return state

    def GetRobotStateStream(self, request, context):
        yield polymetis_pb2.RobotState()

    def GetRobotStateLog(self, request, context):
        yield polymetis_pb2.RobotState()

    def GetEpisodeInterval(self, request, context):
        return polymetis_pb2.LogInterval()

    def InitRobotClient(self, request, context):
        return polymetis_pb2.Empty()

    def ControlUpdate(self, request, context):
        return polymetis_pb2.TorqueCommand()

    def GetRobotClientMetadata(self, request, context):
        """Returns the metadata associated with the robot"""
        return self._metadata.get_proto()


class OriginalPsyonicGripperServer(polymetis_pb2_grpc.GripperServerServicer):
    """gRPC server that exposes Psyonic Ability Hand controls to the client
    Communicates with the gripper through RS-485
    """

    def __init__(self, hand):
        self.hand = hand

    def GetState(self, request, context):
        print("GetState")

        state = polymetis_pb2.GripperState()
        state.timestamp.GetCurrentTime()

        position = np.array(self.hand.position.to_list())
        current = np.array(self.hand.current.to_list())
        touch = np.array([np.array(f) for f in self.hand.touch.to_list()])

        print(f"position: {position}")
        print(f"current: {current}")
        print(f"touch: {touch}")

        avg_position = np.mean(position, dtype=float)
        avg_current = np.mean(current, dtype=float)
        avg_touch = np.mean(touch, dtype=float)

        print(
            f"avg: position: {avg_position} current: {avg_current} touch: {avg_touch}"
        )
        # touch = self.hand.touch

        # width?
        state.width = avg_position
        state.max_width = 100.0
        state.is_grasped = avg_touch > 1000
        state.is_moving = avg_current > 2

        print(f"state: {state}")
        return state

    def Goto(self, request, context):
        return self.Grasp(request, context)

    def Grasp(self, request, context):
        self.hand.grasp(width=request.width, speed=request.speed)
        return polymetis_pb2.Empty()


class GripperServerLauncher:
    def __init__(
        self, comm, metadata_cfg, ip="localhost", port="50052", *args, **kwargs
    ):
        metadata = hydra.utils.instantiate(metadata_cfg)

        self.address = f"{ip}:{port}"

        if comm["type"] == "SerialIO":
            self.hand = Hand(
                SerialIO(port=comm["port"], baud=comm["baud"]), protocol_version=2
            )
        elif comm["type"] == "MockIO":
            self.hand = Hand(MockIO(), protocol_version=2)
        else:
            raise RuntimeError("Unrecognized gripper communication protocol")

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))

        polymetis_pb2_grpc.add_PolymetisControllerServerServicer_to_server(
            PsyonicGripperServer(self.hand, metadata), self.server
        )
        self.server.add_insecure_port(self.address)

    def run(self):
        try:
            self.hand.start()
            self.server.start()
            print(f"Psyonic Ability Hand server running at {self.address}.")
            self.server.wait_for_termination()
        except KeyboardInterrupt:
            pass
        finally:
            pass
            self.hand.stop()
