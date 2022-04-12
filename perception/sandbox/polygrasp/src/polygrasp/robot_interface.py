"""polymetis.RobotInterface combined with GripperInterface, with an additional `grasp` method."""

import polymetis

class GraspingRobotInterface(polymetis.RobotInterface):
    def __init__(self, gripper: polymetis.GripperInterface, *args, **kwargs):
        self.gripper = gripper
    
    def select_grasp(self):
        raise NotImplementedError

    def grasp(self):
        raise NotImplementedError
