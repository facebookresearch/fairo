import torch

import torchcontrol as toco
from torchcontrol.transform import Rotation as R
from polymetis import RobotInterface

from moveit_bridge import MoveitInterface

# Hyperparams
POS_NOISE = torch.Tensor([0.1, 0.1, 0.1])
ORI_NOISE = torch.Tensor([0.25, 0.25, 0.25])
TIME_TO_GO = 3


OBS_MESH_DIR = "../polymetis/polymetis/data/kuka_iiwa/meshes/iiwa7/collision/link_0.stl"
OBJ_MESH_DIR = "../polymetis/polymetis/data/kuka_iiwa/meshes/robotiq-2f/collision/base.stl"


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

    # Add mesh
    print("Adding obstacle...")
    moveit.add_mesh("obstacle1", [0.2, 0, 0.4], [1, 0, 0, 0], OBS_MESH_DIR)
    moveit.attach_mesh("panda_link8", "object1", [0.0, 0, 0.0], [1, 0, 0, 0], OBJ_MESH_DIR)

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
    policy = toco.policies.JointTrajectoryExecutor(
        joint_pos_trajectory=[waypoint["joint_positions"] for waypoint in trajectory],
        joint_vel_trajectory=[waypoint["joint_velocities"] for waypoint in trajectory],
        Kp=robot.metadata.default_Kq,
        Kd=robot.metadata.default_Kqd,
        robot_model=robot.robot_model,
        ignore_gravity=robot.use_grav_comp,
    )

    # Execute plan & evaluate result
    print("Executing policy...")
    robot.send_torch_policy(policy)
    print("Done.")

    ee_pos_final, ee_quat_final = robot.pose_ee()
    print(f"Final pose: pos={ee_pos_final}, quat={ee_quat_final}")

    # Remove objects from scene
    print("Removing objects...")
    moveit.remove_attached_object("panda_link8", "object1")
    moveit.remove_world_object("object1")
    moveit.remove_world_object("obstacle1")
    print("Done.")


if __name__ == "__main__":
    main()
