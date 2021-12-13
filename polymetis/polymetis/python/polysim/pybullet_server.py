# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import sys
from concurrent import futures

import grpc

import polymetis_pb2
import polymetis_pb2_grpc

from .third_party.robotiq_2finger_grippers.robotiq_2f_gripper import (
    Robotiq2FingerGripper,
)


class PolysimServer(polymetis_pb2_grpc.PolymetisControllerServerServicer):
    def __init__(self, cfg, sim_client):
        # Controller stub
        self.channel = grpc.insecure_channel(
            f"{cfg.controller_server.ip}:{cfg.controller_server.port}"
        )
        self.controller_stub = polymetis_pb2_grpc.PolymetisControllerServerStub(
            self.channel
        )

        # Simulation client
        self.sim_client = sim_client

    # Controller Server interface: Forward to polymetis server
    def SetController(self, request, context):
        return self.controller_stub.SetController(request)

    def UpdateController(self, request, context):
        return self.controller_stub.UpdateController(request)

    def TerminateController(self, request, context):
        return self.controller_stub.TerminateController(request)

    def GetRobotState(self, request, context):
        return self.controller_stub.GetRobotState(request)

    def GetRobotStateStream(self, request, context):
        return self.controller_stub.GetRobotStateStream(request)

    def GetRobotStateLog(self, request, context):
        return self.controller_stub.GetRobotStateLog(request)

    def GetEpisodeInterval(self, request, context):
        return self.controller_stub.GetEpisodeInterval(request)

    # Gripper interface
    def GripperGetState(self, request, context):
        pass  # TODO

    def GripperGoto(self, request, context):
        pass  # TODO


def run_server(ip, port):
    """
    sim_client --> controller server
    server --> controller server
    user_client --> server
    """
    # Run client
    sim_client = GrpcSimulationClient(
        metadata_cfg, env, env_cfg, ip, port, log_interval, max_ping
    )
    sim_client.run()

    # Launch simulation server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))

    polymetis_pb2_grpc.add_PolymetisControllerServerServicer_to_server(
        PolysimServer(cfg, sim_client=sim_client), server
    )
    server.add_insecure_port(f"{ip}:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    run_server(sys.argv[1], sys.argv[2])
