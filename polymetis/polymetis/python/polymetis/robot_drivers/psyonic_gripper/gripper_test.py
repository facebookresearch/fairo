# launch_robot.py robot_client=franka_sim use_real_time=false gui=true &
# launch_gripper.py &

import time

from ipaddress import ip_address
from polymetis import GripperInterface


def main():
    gripper = GripperInterface(ip_address="localhost")

    state = gripper.get_state()
    print(f"state: {state}")

    gripper.goto(width=1, speed=0.2, force=0.1)
    time.sleep(7.0)
    state = gripper.get_state()
    print(f"state: {state}")

    gripper.goto(width=0, speed=0.2, force=0.1)
    time.sleep(7.0)
    state = gripper.get_state()
    print(f"state: {state}")

    gripper.goto(width=1, speed=0.8, force=0.1)
    time.sleep(7.0)
    state = gripper.get_state()
    print(f"state: {state}")

    gripper.goto(width=0, speed=0.8, force=0.1)
    time.sleep(7.0)
    state = gripper.get_state()
    print(f"state: {state}")


if __name__ == "__main__":
    main()
