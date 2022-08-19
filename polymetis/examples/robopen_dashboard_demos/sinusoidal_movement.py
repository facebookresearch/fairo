import a0
import polymetis
import numpy as np
import time
import fairomsg
import json

FUNCTION = 1  # 0 = fixed-frequency sinusoid, 1 = chirp

if __name__ == "__main__":
    sensor_msgs = fairomsg.get_msgs("sensor_msgs")
    joint_pub = a0.Publisher("robot/desired_joint_state/sinusoidal")
    json_pub = a0.Publisher("robot/desired_joint_state/sinusoidal_json")

    robot = polymetis.RobotInterface(ip_address="100.97.47.78", enforce_version=False)
    robot.go_home()

    control_hz = 50  # update frequency
    T = 2  # starting period
    time_left = 1000
    chirp = 2500

    target_joint = 4
    joint_limit = 0.5
    joint_positions = robot.get_joint_positions()

    # robot.metadata.default_Kx[1] = 1000 # tuning gains
    robot.start_joint_impedance()

    for i in range(time_left * control_hz):
        start = time.time()

        if FUNCTION == 0:  # fixed-frequency sinusoid
            joint_positions[target_joint] = (
                np.sin(np.pi * i / (T * control_hz)) * joint_limit
            )
        elif FUNCTION == 1:  # chirp function
            joint_positions[target_joint] = (
                np.sin(np.pi * i * i / (T * control_hz * chirp)) * joint_limit
            )

        robot.update_desired_joint_positions(joint_positions)

        a0.update_configs()

        # JointState pub of desired joint state
        ros_msg = sensor_msgs.JointState(
            name=[f"panda_joint{i+1}" for i in range(len(joint_positions))],
            position=joint_positions.tolist(),
        )
        hdrs = [
            ("content-type", 'application/x-capnp; schema="sensor_msgs/JointState"')
        ]
        joint_pub.pub(a0.Packet(hdrs, ros_msg.to_bytes()))

        # JSON pub of desired and actual joint state
        msg = {
            "position": {
                "desired_position": float(joint_positions.numpy()[target_joint]),
                "actual_position": robot.get_robot_state().joint_positions[
                    target_joint
                ],
            }
        }
        hdrs = [("content-type", "application/json")]
        json_pub.pub(a0.Packet(hdrs, json.dumps(msg)))

        end = time.time()
        time.sleep(max(0, (1 / control_hz) - (end - start)))

    robot.terminate_current_policy()
