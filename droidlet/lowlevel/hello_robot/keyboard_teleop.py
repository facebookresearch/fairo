import sys
import time

from pynput import keyboard
import numpy as np


UP = keyboard.Key.up
DOWN = keyboard.Key.down
LEFT = keyboard.Key.left
RIGHT = keyboard.Key.right

VEL_MAX_DEFAULT = 0.15
RVEL_MAX_DEFAULT = 0.9


class RobotController:
    def __init__(
        self,
        mover,
        vel_max=None,
        rvel_max=None,
        hz=10,
    ):
        # Params
        self.dt = 1.0 / hz
        self.vel_max = vel_max or VEL_MAX_DEFAULT
        self.rvel_max = rvel_max or RVEL_MAX_DEFAULT

        # Robot
        print("Connecting to robot...")
        self.robot = mover.bot
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


def run_teleop(mover, vel=None, rvel=None):
    robot_controller = RobotController(mover, vel, rvel)

    listener = keyboard.Listener(
        on_press=robot_controller.on_press,
        on_release=robot_controller.on_release,
    )
    listener.start()

    robot_controller.run()


if __name__ == "__main__":
    from droidlet.lowlevel.hello_robot.hello_robot_mover import HelloRobotMover

    mover = HelloRobotMover(ip=sys.argv[1])
    run_teleop(mover)
