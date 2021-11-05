import torch

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from polymetis import RobotInterface

from moveit_bridge import MoveitInterface

# Hyperparams
POS_NOISE = torch.Tensor([0.1, 0.1, 0.1])
ORI_NOISE = torch.Tensor([0.25, 0.25, 0.25])
TIME_TO_GO = 3


class JointPlan(toco.ControlModule):
    def __init__(self, trajectory):
        super().__init__()

        self.pos_arr = torch.stack([point["joint_positions"] for point in trajectory])
        self.vel_arr = torch.stack([point["joint_velocities"] for point in trajectory])
        self.acc_arr = torch.stack([point["joint_accelerations"] for point in trajectory])

    def forward(self, i: int):
        return self.pos_arr[i, :], self.vel_arr[i, :], self.acc_arr[i, :]


def main():
    # Initialize interfaces
    moveit = MoveitInterface()
    robot = RobotInterface()

    robot.go_home()

    # Sample desired position
    print("\n===== Setup =====")
    ee_pos_current, ee_quat_current = robot.pose_ee()
    print(f"Current pose: pos={ee_pos_current}, quat={ee_quat_current}")

    ee_pos_desired = ee_pos_current + POS_NOISE * torch.randn(3)
    ee_quat_desired = (
        R.from_quat(ee_quat_current) * R.from_rotvec(ORI_NOISE * torch.randn(3))
    ).as_quat()
    print(f"Target pose: pos={ee_pos_desired}, quat={ee_quat_desired}")

    # Plan
    print("\n===== Plan =====")
    print("Planning to target pose...")
    joint_pos_current = robot.get_joint_angles()
    trajectory, info = moveit.plan(
        joint_pos_current,
        ee_pos_desired,
        ee_quat_desired,
        ee_link=robot.metadata.ee_link_name,
        time_to_go=TIME_TO_GO,
        hz=robot.metadata.hz,
    )

    # Plan failed
    if not info["success"]:
        print("Planning failed!")
        print(f"info:{info}")
        return

    print("Planning succeeded!")

    # Create plan policy
    print("\n===== Policy =====")
    print("Creating plan policy...")
    plan = JointPlan(trajectory)
    policy = toco.policies.JointPlanExecutor(
        plan=plan,
        robot_model=robot.robot_model,
        steps=len(trajectory),
        Kp=robot.metadata.default_Kq,
        Kd=robot.metadata.default_Kqd,
    )

    # Execute plan & evaluate result
    print("Executing policy...")
    robot.send_torch_policy(policy)
    print("Done.")

    ee_pos_final, ee_quat_final = robot.pose_ee()
    print(f"Final pose: pos={ee_pos_final}, quat={ee_quat_final}")


if __name__ == "__main__":
    main()
