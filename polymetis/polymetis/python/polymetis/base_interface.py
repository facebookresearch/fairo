# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import io
from typing import Dict, Generator, List, Callable
import time
import threading
import atexit
import logging

import grpc  # This requires `conda install grpcio protobuf`
import torch

import polymetis
from polymetis_pb2 import LogInterval, RobotState, ControllerChunk, Empty
from polymetis_pb2_grpc import PolymetisControllerServerStub
import torchcontrol as toco

log = logging.getLogger(__name__)


# Maximum bytes we send per message to server (so as not to overload it).
MAX_BYTES_PER_MSG = 1024

# Polling rate when waiting for episode to finish
POLLING_RATE = 50

# Grpc empty object
EMPTY = Empty()


# Dict container as a nn.module to enable use of jit.save & jit.load
class ParamDictContainer(torch.nn.Module):
    """A torch.nn.Module container for a parameter key.

    Note:
        This is necessary because TorchScript can only script modules,
        not tensors or dictionaries.

    Args:
        param_dict: The dictionary mapping parameter names to values.
    """

    param_dict: Dict[str, torch.Tensor]

    def __init__(self, param_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self.param_dict = param_dict

    def forward(self) -> Dict[str, torch.Tensor]:
        """Simply returns the wrapped parameter dictionary."""
        return self.param_dict


class BaseRobotInterface:
    """Base robot interface class to initialize a connection to a gRPC controller manager server.

    Args:
        ip_address: IP address of the gRPC-based controller manager server.
        port: Port to connect to on the IP address.
    """

    def __init__(
        self, ip_address: str = "localhost", port: int = 50051, enforce_version=True
    ):
        # Create connection
        self.channel = grpc.insecure_channel(f"{ip_address}:{port}")
        self.grpc_connection = PolymetisControllerServerStub(self.channel)

        # Get metadata
        self.metadata = self.grpc_connection.GetRobotClientMetadata(EMPTY)

        # Check version
        if enforce_version:
            client_ver = polymetis.__version__
            server_ver = self.metadata.polymetis_version
            assert (
                client_ver == server_ver
            ), "Version mismatch between client & server detected! Set enforce_version=False to bypass this error."

    def __del__(self):
        # Close connection in destructor
        self.channel.close()

    @staticmethod
    def _get_msg_generator(scripted_module) -> Generator:
        """Given a scripted module, return a generator of its serialized bits
        as byte chunks of max size MAX_BYTES_PER_MSG."""
        # Write into bytes buffer
        buffer = io.BytesIO()
        torch.jit.save(scripted_module, buffer)
        buffer.seek(0)

        # Create policy generator
        def msg_generator():
            # A generator which chunks a scripted module into messages of
            # size MAX_BYTES_PER_MSG and send these messages to the server.
            while True:
                chunk = buffer.read(MAX_BYTES_PER_MSG)
                if not chunk:  # end of buffer
                    break
                msg = ControllerChunk(torchscript_binary_chunk=chunk)
                yield msg

        return msg_generator

    def _get_robot_state_log(
        self, log_interval: LogInterval, timeout: float = None
    ) -> List[RobotState]:
        """A private helper method to get the states corresponding to a log_interval from the server.

        Args:
            log_interval: a message holding start and end indices for a trajectory of RobotStates.
            timeout: Amount of time (in seconds) to wait before throwing a TimeoutError.

        Returns:
            If successful, returns a list of RobotState objects.

        """
        robot_state_generator = self.grpc_connection.GetRobotStateLog(log_interval)

        def cancel_rpc():
            log.info("Cancelling attempt to get robot state log.")
            robot_state_generator.cancel()
            log.info(f"Cancellation completed.")

        atexit.register(cancel_rpc)

        results = []

        def read_stream():
            try:
                for state in robot_state_generator:
                    results.append(state)
            except grpc.RpcError as e:
                log.error(f"Unable to read stream of robot states: {e}")

        read_thread = threading.Thread(target=read_stream)
        read_thread.start()
        read_thread.join(timeout=timeout)

        if read_thread.is_alive():
            raise TimeoutError("Operation timed out.")
        else:
            atexit.unregister(cancel_rpc)
            return results

    def get_robot_state(self) -> RobotState:
        """Returns the latest RobotState."""
        return self.grpc_connection.GetRobotState(EMPTY)

    def get_previous_interval(self, timeout: float = None) -> LogInterval:
        """Get the log indices associated with the currently running policy."""
        log_interval = self.grpc_connection.GetEpisodeInterval(EMPTY)
        assert log_interval.start != -1, "Cannot find previous episode."
        return log_interval

    def is_running_policy(self) -> bool:
        log_interval = self.grpc_connection.GetEpisodeInterval(EMPTY)
        return (
            log_interval.start != -1  # policy has started
            and log_interval.end == -1  # policy has not ended
        )

    def get_previous_log(self, timeout: float = None) -> List[RobotState]:
        """Get the list of RobotStates associated with the currently running policy.

        Args:
            timeout: Amount of time (in seconds) to wait before throwing a TimeoutError.

        Returns:
            If successful, returns a list of RobotState objects.

        """
        log_interval = self.get_previous_interval(timeout)
        return self._get_robot_state_log(log_interval, timeout=timeout)

    def send_torch_policy(
        self,
        torch_policy: toco.PolicyModule,
        blocking: bool = True,
        timeout: float = None,
        post_exe_hook: Callable = None,
    ) -> List[RobotState]:
        """Sends the ScriptableTorchPolicy to the server.

        Args:
            torch_policy: An instance of ScriptableTorchPolicy to control the robot.
            blocking: If True, blocks until the policy is finished executing, then returns the list of RobotStates.
            timeout: Amount of time (in seconds) to wait before throwing a TimeoutError.

        Returns:
            If `blocking`, returns a list of RobotState objects. Otherwise, returns None.

        """
        start_time = time.time()

        # Script & chunk policy
        scripted_policy = torch.jit.script(torch_policy)
        msg_generator = self._get_msg_generator(scripted_policy)

        # Send policy as stream
        try:
            log_interval = self.grpc_connection.SetController(msg_generator())
        except grpc.RpcError as e:
            raise grpc.RpcError(f"POLYMETIS SERVER ERROR --\n{e.details()}") from None

        if blocking:
            # Check policy termination
            while log_interval.end == -1:
                log_interval = self.grpc_connection.GetEpisodeInterval(EMPTY)

                if timeout is not None and time.time() - start_time > timeout:
                    raise TimeoutError("Operation timed out.")
                time.sleep(1.0 / POLLING_RATE)

            # Execute post-execution hook
            if post_exe_hook is not None:
                post_exe_hook()

            # Retrieve robot state log
            if timeout is not None:
                time_passed = time.time() - start_time
                timeout = timeout - time_passed
            return self._get_robot_state_log(log_interval, timeout=timeout)

    def update_current_policy(self, param_dict: Dict[str, torch.Tensor]) -> int:
        """Updates the current policy's with a (possibly incomplete) dictionary holding the updated values.

        Args:
            param_dict: A dictionary mapping from param_name to updated torch.Tensor values.

        Returns:
            Index offset from the beginning of the episode when the update was applied.

        """
        # Script & chunk params
        scripted_params = torch.jit.script(ParamDictContainer(param_dict))
        msg_generator = self._get_msg_generator(scripted_params)

        # Send params container as stream
        try:
            update_interval = self.grpc_connection.UpdateController(msg_generator())
        except grpc.RpcError as e:
            raise grpc.RpcError(f"POLYMETIS SERVER ERROR --\n{e.details()}") from None
        episode_interval = self.grpc_connection.GetEpisodeInterval(EMPTY)

        return update_interval.start - episode_interval.start

    def terminate_current_policy(
        self, return_log: bool = True, timeout: float = None
    ) -> List[RobotState]:
        """Terminates the currently running policy and (optionally) return its trajectory.

        Args:
            return_log: whether or not to block & return the policy's trajectory.
            timeout: Amount of time (in seconds) to wait before throwing a TimeoutError.

        Returns:
            If `return_log`, returns the list of RobotStates the list of RobotStates corresponding to the current policy's execution.

        """
        # Send termination
        log_interval = self.grpc_connection.TerminateController(EMPTY)

        # Query episode log
        if return_log:
            return self._get_robot_state_log(log_interval, timeout=timeout)