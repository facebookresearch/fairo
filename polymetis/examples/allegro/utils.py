from typing import Dict, List
from polymetis.robot_interface import RobotInterface
from polymetis_pb2.polymetis_pb2 import RobotState
import torch
import torchcontrol as toco
import logging

log = logging.getLogger(__name__)


class ExtRobotInterface(RobotInterface):
    def __init__(
        self, time_to_go_default: float = 3, use_grav_comp: bool = True, *args, **kwargs
    ):
        super().__init__(time_to_go_default, use_grav_comp, *args, **kwargs)

    def move_to_joint_positions(
        self,
        positions: torch.Tensor,
        time_to_go: float = None,
        time_to_hold: float = 0,
        delta: bool = False,
        Kq: torch.Tensor = None,
        Kqd: torch.Tensor = None,
        **kwargs,
    ) -> List[RobotState]:

        # Parse parameters
        joint_pos_current = self.get_joint_positions()
        joint_pos_desired = torch.Tensor(positions)
        if delta:
            joint_pos_desired += joint_pos_current

        time_to_go_adaptive = self._adaptive_time_to_go(
            joint_pos_desired - joint_pos_current
        )
        if time_to_go is None:
            time_to_go = time_to_go_adaptive
        elif time_to_go < time_to_go_adaptive:
            log.warn(
                "The specified 'time_to_go' might not be large enough to ensure accurate movement."
            )

        # Plan trajectory
        waypoints = toco.planning.generate_joint_space_min_jerk(
            start=joint_pos_current,
            goal=joint_pos_desired,
            time_to_go=time_to_go,
            hz=self.hz,
        )

        last_waypoint = waypoints[-1]
        for t in torch.arange(time_to_go, time_to_hold + time_to_go, 1 / self.hz):
            waypoint = last_waypoint.copy()
            waypoint["time_from_start"] = t
            waypoints.append(waypoint)

        if Kq is None:
            Kq = self.Kq_default
        if Kqd is None:
            Kqd = self.Kqd_default
        # Create & execute policy
        torch_policy = toco.policies.JointTrajectoryExecutor(
            joint_pos_trajectory=[waypoint["position"] for waypoint in waypoints],
            joint_vel_trajectory=[waypoint["velocity"] for waypoint in waypoints],
            Kp=Kq,
            Kd=Kqd,
            robot_model=self.robot_model,
            ignore_gravity=self.use_grav_comp,
        )

        return self.send_torch_policy(torch_policy=torch_policy, **kwargs)


class MyPDPolicy(toco.PolicyModule):
    """
    Custom policy that performs PD control around a desired joint position
    """

    def __init__(self, joint_pos_current, kq, kqd, **kwargs):
        """
        Args:
            joint_pos_current (torch.Tensor):   Joint positions at initialization
            kq, kqd (torch.Tensor):             PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.q_desired = torch.nn.Parameter(joint_pos_current)

        # Initialize modules
        self.feedback = toco.modules.JointSpacePD(kq, kqd)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]

        # Execute PD control
        output = self.feedback(
            q_current, qd_current, self.q_desired, torch.zeros_like(qd_current)
        )

        return {"joint_torques": output}
