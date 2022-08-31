import sys
import time

from pynput import keyboard
import numpy as np


UP = keyboard.Key.up
DOWN = keyboard.Key.down
LEFT = keyboard.Key.left
RIGHT = keyboard.Key.right


class RobotController:
    def __init__(
        self,
        mover,
        hz=10,
        vel_max=0.25,
        rvel_max=0.5,
    ):
        # Params
        self.dt = 1.0 / hz
        self.vel_max = vel_max
        self.rvel_max = rvel_max

        # Robot
        print("Connecting to robot...")
        self.robot = mover
        print("Connected.")

        # Keyboard
        self.key_states = {key: 0 for key in [UP, DOWN, LEFT, RIGHT]}

        # Controller states
        self.vel = 0
        self.rvel = 0

    def on_press(self, key):
        self.key_states[key] = 1

    def on_release(self, key):
        self.key_states[key] = 0

    def run(self):
        print("Teleoperation started.")
        while True:
            # Map keystrokes
            vert_sign = self.key_states[UP] - self.key_states[DOWN]
            hori_sign = self.key_states[LEFT] - self.key_states[RIGHT]

            # Compute velocity commands
            self.vel = self.vel_max * vert_sign
            self.rvel = self.rvel_max * hori_sign

            # Command robot
            self.robot.set_velocity(self.vel, self.rvel)

            # Spin
            time.sleep(self.dt)


def run_teleop(mover):
    robot_controller = RobotController(mover)

    listener = keyboard.Listener(
        on_press=robot_controller.on_press,
        on_release=robot_controller.on_release,
    )
    listener.start()

    robot_controller.run()


def run_teleop_from_ip(ip):
    from droidlet.lowlevel.hello_robot.hello_robot_mover import HelloRobotMover

    mover = HelloRobotMover(ip=ip)
    run_teleop(mover)


if __name__ == "__main__":
    run_teleop(sys.argv[1])
