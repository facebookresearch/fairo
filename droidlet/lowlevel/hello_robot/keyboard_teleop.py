import time

from pynput import keyboard 
import numpy as np

from droidlet.hello_robot.hello_robot_mover import HelloRobotMover

UP = keyboard.Key.up
DOWN = keyboard.Key.down
LEFT = keyboard.Key.left
RIGHT = keyboard.Key.right


class RobotController:
    def __init__(self,
        ip,
        hz=10,
        acc=0.1,
        racc=0.1,
        vel_max=1.,
        rvel_max=1.,
    ):
        # Params
        self.dt = 1 / hz.
        self.acc = acc
        self.racc = racc
        self.vel_max = vel_max
        self.rvel_max = rvel_max

        # Robot
        self.robot = HelloRobotMover(ip=ip)

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
        while True:
            # Map keystrokes
            acc = self.acc * (self.key_states[UP] - self.key_states[DOWN])
            racc = self.racc * (self.key_states[LEFT] - self.key_states[RIGHT])

            # Compute velocity commands
            self.vel = np.clip(self.vel + acc * self.dt, -self.vel_max, self.vel_max)
            self.rvel = np.clip(self.rvel + racc * self.dt, -self.rvel_max, self.rvel_max)

            # Command robot
            self.robot.set_vel(self.vel, self.rvel)

            # Spin
            time.sleep(self.dt)


if __name__ == '__main__':
    robot_controller = RobotController()

    listener = keyboard.Listener(
        on_press=robot_controller.on_press,
        on_release=robot_controller.on_release,
    )
    listener.start()
    
    robot_controller.run()
