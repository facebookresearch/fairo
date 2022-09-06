#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable
import time
from threading import Thread
import numpy as np
import hydra
from omegaconf.dictconfig import DictConfig

import grpc
import polymetis_pb2
import polymetis_pb2_grpc

from polymetis import RobotInterface
from polymetis.utils import Spinner
from polymetis.robot_client.abstract_robot_client import (
    AbstractRobotClient,
)

from polysim.envs import AbstractControlledEnv

import logging

log = logging.getLogger(__name__)


class GrpcSimulationClient(AbstractRobotClient):
    """A RobotClient which wraps a PyBullet simulation.

    Args:
        metadata_cfg: A Hydra config which sepcifies the metadata required to initialize
                      a RobotClient with the server.

        env: A simulation environment implementing AbstractControlledEnv.

        env_cfg: If `env` is not passed, this is a Hydra config which specifies an `env`
                 to instantiate an equivalent env.

        ip: Server IP.

        log_interval: Log every `log_interval` number of timesteps. 0 if no logging.

        max_ping: The amount of time in seconds; if a request takes long than this,
                  send a debug message warning.

    """

    def __init__(
        self,
        metadata_cfg: DictConfig = None,
        env: AbstractControlledEnv = None,
        env_cfg: DictConfig = None,
        ip: str = "localhost",
        port: int = 50051,
        log_interval: int = 0,
        max_ping: float = 0.0,
        mirror_hz: float = 24.0,
    ):
        super().__init__(metadata_cfg=metadata_cfg)

        # Simulation env
        if env is not None:
            assert isinstance(
                env, AbstractControlledEnv
            ), "'env' argument must be an instance of 'AbstractControlledEnv'"
            self.env = env
        elif env_cfg is not None:
            assert isinstance(
                env_cfg, DictConfig
            ), "'env_cfg' argument must be a Hydra config (omegaconf.dictconfig.DictConfig)"
            self.env = hydra.utils.instantiate(env_cfg)
        else:
            raise Exception(
                "No env specified. Either 'env' or 'env_cfg' input argument must be specified"
            )
        self.env.reset()

        # GRPC connection
        self.channel = grpc.insecure_channel(f"{ip}:{port}")
        self.connection = polymetis_pb2_grpc.PolymetisControllerServerStub(self.channel)

        # Loop time
        self.hz = self.metadata.get_proto().hz

        # Round trip time logging:
        # If log_interval > 0, store the last log_interval
        # round-trip times in interval log and print when full
        self.log_interval = log_interval
        self.max_ping = max_ping
        self.i = 0
        self.interval_log = []
        self.round_trip_time_buffer = 0.0

        self._state_setter = None
        self._kill_state_setter = False

        self._runner = None
        self._kill_runner = False

        self.mirror_hz = mirror_hz

    def __del__(self):
        """Close connection in destructor"""
        self.channel.close()
        if self._state_setter:
            self.unsync()

    def run_no_wait(self, time_horizon=float("inf")):
        assert self._runner is None, "Simulator already running in background thread!"
        self._runner = Thread(target=self.run, args=[time_horizon, True], daemon=True)
        self._runner.start()

    def kill_run(self):
        assert self._runner is not None, "No background simulator to kill!"
        self._kill_runner = True
        self._runner.join()
        self._runner = None

    def run(self, time_horizon=float("inf"), threaded=False):
        """Start running the simulation and querying the server.

        Args:
            time_horizon: If finite, the number of timesteps to stop the simulation.

        """
        msg = self.connection.InitRobotClient(self.metadata.get_proto())

        robot_state = polymetis_pb2.RobotState()
        # Main loop
        t = 0
        spinner = Spinner(self.hz)
        while t < time_horizon:
            if threaded:
                # print(f"Threaded run {t}")
                if self._kill_runner:
                    break
            # Get robot state from env
            joint_pos, joint_vel = self.env.get_current_joint_pos_vel()
            robot_state.joint_positions[:] = joint_pos
            robot_state.joint_velocities[:] = joint_vel

            (
                torques_commanded,
                torques_applied,
                torques_measured,
                torques_external,
            ) = self.env.get_current_joint_torques()
            robot_state.prev_joint_torques_computed[:] = torques_commanded
            robot_state.prev_joint_torques_computed_safened[:] = torques_applied
            robot_state.motor_torques_measured[:] = torques_measured
            robot_state.motor_torques_external[:] = torques_external

            robot_state.timestamp.GetCurrentTime()
            robot_state.prev_controller_latency_ms = self.round_trip_time_buffer
            robot_state.prev_command_successful = True
            robot_state.error_code = 0

            # Query controller manager server for action
            # TODO: have option for async mode through async calls, see
            # https://grpc.io/docs/languages/python/basics/#simple-rpc-1
            log_request_time = self.log_interval > 0 and t % self.log_interval == 0
            msg = self.execute_rpc_call(
                self.connection.ControlUpdate,
                [robot_state],
                log_request_time=log_request_time,
            )

            # Apply action to env
            torque_command = np.array([t for t in msg.joint_torques])
            self.env.apply_joint_torques(torque_command)

            # Idle for the remainder of loop time
            t += 1
            spinner.spin()

    def execute_rpc_call(self, request_func: Callable, args=[], log_request_time=False):
        """Executes an RPC call and performs round trip time intervals checks and logging

        Args:
            request_func: A Python function to call an RPC.

            args: List of arguments to pass to `request_func`.

            log_request_time: Whether to log the debug messages.

        """
        # Execute RPC call
        prev_time = time.time_ns()
        ret = request_func(*args)
        round_trip_time = (time.time_ns() - prev_time) / 1000.0 / 1000.0

        # Check round trip time
        if self.max_ping > 0.0 and round_trip_time > self.max_ping:
            log.debug(
                f"\n==== Warning: round trip time takes {round_trip_time} ms! ====\n"
            )

        # Log round trip times
        self.round_trip_time_buffer = round_trip_time
        if self.log_interval > 0:
            self.interval_log.append(round_trip_time)
            if log_request_time:
                log.debug(
                    f"\nTime per RPC request in ms: avg: {sum(self.interval_log) / len(self.interval_log)} max: {max(self.interval_log)} min: {min(self.interval_log)}"
                )
                self.interval_log = []

        return ret

    def init_robot_client(self):
        self.connection.InitRobotClient(self.metadata.get_proto())

    def set_robot_state(self, robot_state: polymetis_pb2.RobotState):
        self.env.set_robot_state(robot_state)

    def _sync_blocking(self, tgt_robot: RobotInterface, timesteps: int):
        step = 0
        spinner = Spinner(self.mirror_hz)
        while (timesteps < 0 or step < timesteps) and not self._kill_state_setter:
            tgt_state = tgt_robot.get_robot_state()
            self.set_robot_state(tgt_state)
            step += 1
            spinner.spin()

    def sync(self, tgt_robot: RobotInterface, timesteps: int = -1):
        assert (
            self._state_setter is None
        ), "The simulation client is already synced to a robot"
        self._state_setter = Thread(
            target=self._sync_blocking, args=[tgt_robot, timesteps], daemon=True
        )
        self._state_setter.start()

    def unsync(self):
        assert self._state_setter is not None, "The mirror simulator is not synced"
        self._kill_state_setter = True
        self._state_setter.join()
        self._state_setter = None
