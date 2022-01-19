from polymetis.utils.continuous_grasper import ManipulatorSystem


def track_num_successes_per_100():
    iters = 100
    robot_kwargs = {}
    gripper_kwargs = {}
    robot = ManipulatorSystem(robot_kwargs, gripper_kwargs)
    total_successes, total_tries = robot.continuously_grasp(iters)
    return total_successes


track_num_successes_per_100.unit = "num_successes"
