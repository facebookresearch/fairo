import a0
import polymetis
import torch
import time
import fairomsg
import json

if __name__ == "__main__":
    std_msgs = fairomsg.get_msgs("std_msgs")
    geometry_msgs = fairomsg.get_msgs("geometry_msgs")
    transform_pub = a0.Publisher("robot/desired_ee_pose/square")
    json_pub = a0.Publisher("robot/desired_ee_pose/square_json")

    robot = polymetis.RobotInterface(ip_address="100.97.47.78", enforce_version=False)
    robot.go_home()

    ee_pos, ee_quat = robot.get_ee_pose()

    # Move to starting position
    starting_position = -0.1
    ee_pos += torch.Tensor([0.0, starting_position, starting_position])
    robot.move_to_ee_pose(ee_pos)

    y_ee_limit = 0.45
    z_ee_limit = 0.7
    side_length = min(y_ee_limit - ee_pos[1], z_ee_limit - ee_pos[2])

    control_hz = 50
    time_left = 10
    steps_per_side = control_hz * time_left / 4
    ee_step = side_length / steps_per_side

    robot.start_cartesian_impedance()

    for i in range(control_hz * time_left):
        a0.update_configs()
        start = time.time()

        # Update ee pose
        if i < steps_per_side:
            movement = [0.0, ee_step, 0.0]
        elif i < steps_per_side * 2:
            movement = [0.0, 0.0, ee_step]
        elif i < steps_per_side * 3:
            movement = [0.0, -ee_step, 0.0]
        else:
            movement = [0.0, 0.0, -ee_step]

        ee_pos += torch.Tensor(movement)
        robot.update_desired_ee_pose(ee_pos)

        # TransformStamped pub of ee pose
        ee_pos_list = ee_pos.tolist()
        ee_quat_list = ee_quat.tolist()
        ros_msg = geometry_msgs.TransformStamped(
            header=std_msgs.Header(frameId="robot_base"),
            childFrameId=f"robot_end_effector",
            transform=geometry_msgs.Transform(
                translation=geometry_msgs.Vector3(
                    x=ee_pos_list[0], y=ee_pos_list[1], z=ee_pos_list[2]
                ),
                rotation=geometry_msgs.Quaternion(
                    x=ee_quat_list[0],
                    y=ee_quat_list[1],
                    z=ee_quat_list[2],
                    w=ee_quat_list[3],
                ),
            ),
        )
        hdrs = [
            (
                "content-type",
                'application/x-capnp; schema="geometry_msgs/TransformStamped"',
            )
        ]
        transform_pub.pub(a0.Packet(hdrs, ros_msg.to_bytes()))

        # JSON pub of desired and actual joint state
        new_ee_pos = robot.get_ee_pose()[0].tolist()
        msg = {
            "position": {
                "desired_x": ee_pos_list[0],
                "desired_y": ee_pos_list[1],
                "desired_z": ee_pos_list[2],
                "actual_x": new_ee_pos[0],
                "actual_y": new_ee_pos[1],
                "actual_z": new_ee_pos[2],
            }
        }
        hdrs = [("content-type", "application/json")]
        json_pub.pub(a0.Packet(hdrs, json.dumps(msg)))

        end = time.time()
        time.sleep(max(0, (1 / control_hz) - (end - start)))

    robot.terminate_current_policy()
