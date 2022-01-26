import numpy as np

from polymetis import RobotInterface


def output_episode_stats(episode_name, robot_states):
    latency_arr = np.array(
        [robot_state.prev_controller_latency_ms for robot_state in robot_states]
    )
    latency_mean = np.mean(latency_arr)
    latency_std = np.std(latency_arr)
    latency_max = np.max(latency_arr)
    latency_min = np.min(latency_arr)

    print(
        f"{episode_name}: {latency_mean:.4f}/ {latency_std:.4f} / {latency_max:.4f} / {latency_min:.4f}"
    )


if __name__ == "__main__":
    robot = RobotInterface()

    print("Control loop latency stats in milliseconds (avg / std / max / min): ")

    # Test joint PD
    robot_states = robot.set_joint_positions(robot.get_joint_angles())
    output_episode_stats("Joint PD", robot_states)

    # Test cartesian PD
    robot_states = robot.set_ee_pose(robot.pose_ee()[0])
    output_episode_stats("Cartesian PD", robot_states)
