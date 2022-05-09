import getch

import numpy as np
import sophus as sp

from .base import TeleopDeviceReader


class KeyboardReader(TeleopDeviceReader):
    """Allows for teleoperation using the keyboard
    Control end-effector position with WASD and RF, toggle gripper state with space
    """

    def __init__(self):
        self.steps = 0
        self.delta_pos = np.zeros(3)
        self.delta_rot = np.zeros(3)
        self.grasp_state = 0

        print("Keyboard teleop reader instantiated.")

    def get_state(self):
        # Get data from keyboard
        key = getch.getch()
        if key == "w":  # Translation
            self.delta_pos[0] += 0.01
        elif key == "s":
            self.delta_pos[0] -= 0.01
        elif key == "a":
            self.delta_pos[1] += 0.01
        elif key == "d":
            self.delta_pos[1] -= 0.01
        elif key == "r":
            self.delta_pos[2] += 0.01
        elif key == "f":
            self.delta_pos[2] -= 0.01
        elif key == "z":  # Rotation
            self.delta_rot[0] += 0.05
        elif key == "Z":
            self.delta_rot[0] -= 0.05
        elif key == "x":
            self.delta_rot[1] += 0.05
        elif key == "X":
            self.delta_rot[1] -= 0.05
        elif key == "c":
            self.delta_rot[2] += 0.05
        elif key == "C":
            self.delta_rot[2] -= 0.05
        elif key == " ":  # Gripper toggle
            self.grasp_state = 1 - self.grasp_state

        self.steps += 1

        # Generate output
        is_active = True

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = sp.SO3.exp(self.delta_rot).matrix() @ pose_matrix[:3, :3]
        pose_matrix[:3, -1] = self.delta_pos
        pose = sp.SE3(pose_matrix)

        grasp_state = self.grasp_state

        return is_active, pose, grasp_state
