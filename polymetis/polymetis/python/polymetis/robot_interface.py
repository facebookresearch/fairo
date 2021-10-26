# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import io
import signal
from typing import Dict, Generator, List, Tuple
import time
import tempfile
import threading

import grpc  # This requires `conda install grpcio protobuf`
import torch

from polymetis_pb2 import LogInterval, RobotState, ControllerChunk, Empty
from polymetis_pb2_grpc import PolymetisControllerServerStub

import torchcontrol as toco


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
        signal.signal(signal.SIGINT, lambda x, y: robot_state_generator.cancel())

        results = []

        def read_stream():
            try:
                for state in robot_state_generator:
                    results.append(state)
            except grpc.RpcError as exc:
                print(exc)

        read_thread = threading.Thread(target=read_stream)
        read_thread.start()
        read_thread.join(timeout=timeout)

        if read_thread.is_alive():
            raise TimeoutError("Operation timed out.")
        else:
            return results

    def get_robot_state(self) -> RobotState:
        """Returns the latest RobotState."""
        return self.grpc_connection.GetRobotState(EMPTY)

    def get_previous_interval(self, timeout: float = None) -> LogInterval:
        """Get the log indices associated with the currently running policy."""
        log_interval = self.grpc_connection.GetEpisodeInterval(EMPTY)
        assert log_interval.start != -1, "Cannot find previous episode."
        return log_interval

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
            print(e)
            return

        if blocking:
            # Check policy termination
            while log_interval.end == -1:
                log_interval = self.grpc_connection.GetEpisodeInterval(EMPTY)

                if timeout is not None and time.time() - start_time > timeout:
                    raise TimeoutError("Operation timed out.")
                time.sleep(1.0 / POLLING_RATE)

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
            print(e)
            return

        episode_interval = self.grpc_connection.GetEpisodeInterval(EMPTY)

        return update_interval.start - episode_interval.start

    def terminate_current_policy(
        self, return_log: LogInterval = True, timeout: float = None
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


class RobotInterface(BaseRobotInterface):
    """
    Adds user-friendly helper methods to automatically construct some policies
    with sane defaults.

    Args:
        time_to_go_default: Default amount of time for policies to run, if not given.

        use_grav_comp: If True, assumes that gravity compensation torques are added
                       to the given torques.

    """

    def __init__(
        self,
        time_to_go_default: float = 4.0,
        use_grav_comp: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        with tempfile.NamedTemporaryFile("w+") as urdf_file:
            urdf_file.write(self.metadata.urdf_file)
            self.set_robot_model(urdf_file.name, self.metadata.ee_link_name)

        self.set_home_pose(torch.Tensor(self.metadata.rest_pose))

        self.Kq_default = torch.Tensor(self.metadata.default_Kq)
        self.Kqd_default = torch.Tensor(self.metadata.default_Kqd)
        self.hz = self.metadata.hz

        self.time_to_go_default = time_to_go_default

        self.use_grav_comp = use_grav_comp

    """
    Setter methods
    """

    def set_home_pose(self, home_pose: torch.Tensor):
        """Sets the home pose for `go_home()` to use."""
        self.home_pose = home_pose

    def set_robot_model(self, robot_description_path: str, ee_link_name: str):
        """Loads the URDF as a RobotModelPinocchio."""
        # Create Torchscript Pinocchio model for DynamicsControllers
        self.robot_model = toco.models.RobotModelPinocchio(
            robot_description_path, ee_link_name
        )

    """
    Getter methods
    """

    def get_joint_angles(self) -> torch.Tensor:
        return torch.Tensor(self.get_robot_state().joint_positions)

    def get_joint_velocities(self) -> torch.Tensor:
        return torch.Tensor(self.get_robot_state().joint_velocities)

    """
    End-effector computation methods
    """

    def pose_ee(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes forward kinematics on the current joint angles.

        Returns:
            torch.Tensor: 3D end-effector position
            torch.Tensor: 4D end-effector orientation as quaternion
        """
        joint_pos = self.get_joint_angles()
        pos, quat = self.robot_model.forward_kinematics(joint_pos)
        return pos, quat

    def get_jacobian(joint_angles):
        raise NotImplementedError  # TODO

    """
    Movement methods
    """

    def set_joint_positions(
        self,
        desired_positions: torch.Tensor,
        time_to_go: float = None,
        Kq: torch.Tensor = None,
        Kqd: torch.Tensor = None,
        **kwargs,
    ) -> List[RobotState]:
        """Uses JointGoToPolicy to move to the desired positions with the given gains."""
        assert (
            self.robot_model is not None
        ), "Robot model not assigned! Call 'set_robot_model(<path_to_urdf>, <ee_link_name>)' to enable use of dynamics controllers"

        if time_to_go is None:
            time_to_go = self.time_to_go_default
        if Kq is None:
            Kq = self.Kq_default
        if Kqd is None:
            Kqd = self.Kqd_default

        joint_pos_current = self.get_joint_angles()
        torch_policy = toco.policies.JointSpaceMoveTo(
            joint_pos_current=joint_pos_current,
            joint_pos_desired=desired_positions,
            time_to_go=time_to_go,
            Kp=Kq,
            Kd=Kqd,
            robot_model=self.robot_model,
            hz=self.hz,
            ignore_gravity=self.use_grav_comp,
        )
        return self.send_torch_policy(torch_policy=torch_policy, **kwargs)

    def move_joint_positions(
        self, delta_positions, *args, **kwargs
    ) -> List[RobotState]:
        """Calls set_joint_positions by adding delta_positions to the curent positions."""
        desired_positions = self.get_joint_angles() + torch.Tensor(delta_positions)
        return self.set_joint_positions(desired_positions, *args, **kwargs)

    def go_home(self, *args, **kwargs) -> List[RobotState]:
        """Calls set_joint_positions to the current home positions."""
        assert (
            self.home_pose is not None
        ), "Home pose not assigned! Call 'set_home_pose(<joint_angles>)' to enable homing"
        return self.set_joint_positions(
            desired_positions=self.home_pose, *args, **kwargs
        )

    def set_ee_pose(
        self,
        position: torch.Tensor,
        orientation: torch.Tensor = None,
        time_to_go: float = None,
        Kq: torch.Tensor = None,
        Kqd: torch.Tensor = None,
        **kwargs,
    ) -> List[RobotState]:
        """Uses OperationalSpaceController to move to a desired end-effector position (and, optionally orientation)."""
        assert (
            self.robot_model is not None
        ), "Robot model not assigned! Call 'set_robot_model(<path_to_urdf>, <ee_link_name>)' to enable use of dynamics controllers"

        if time_to_go is None:
            time_to_go = self.time_to_go_default
        if Kq is None:
            Kq = self.Kq_default
        if Kqd is None:
            Kqd = self.Kqd_default

        joint_pos_current = self.get_joint_angles()
        torch_policy = toco.policies.OperationalSpaceMoveTo(
            joint_pos_current=joint_pos_current,
            ee_pos_desired=position,
            ee_quat_desired=orientation,
            time_to_go=time_to_go,
            Kp=Kq,
            Kd=Kqd,
            robot_model=self.robot_model,
            hz=self.hz,
            ignore_gravity=self.use_grav_comp,
        )
        return self.send_torch_policy(torch_policy=torch_policy, **kwargs)

    def move_ee_xyz(
        self, displacement: torch.Tensor, use_orient: bool = True, **kwargs
    ) -> List[RobotState]:
        """Moves to a desired end-effector position by adding `displacement` to the current end effector position.

        Args:
            displacement: 3D delta end-effector position.
            use_orient: Use the current end-effector orientation as the desired orientation.
        """
        ee_pos, ee_orient = self.pose_ee()
        if not use_orient:
            ee_orient = None
        target_ee_pos = ee_pos + torch.Tensor(displacement)
        return self.set_ee_pose(target_ee_pos, ee_orient, **kwargs)
