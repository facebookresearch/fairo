import numpy as np
from scipy.spatial.transform import Rotation

def transform_global_to_base(XYT, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : base position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    XYT = np.asarray(XYT)
    new_T = XYT[2] - current_pose[2]
    R = Rotation.from_euler("Z", current_pose[2]).as_matrix()
    print(R)
    XYT[0] = XYT[0] - current_pose[0]
    XYT[1] = XYT[1] - current_pose[1]
    out_XYT = np.matmul(XYT.reshape(-1, 3), R).reshape((-1, 3))
    out_XYT = out_XYT.ravel()
    return [out_XYT[0], out_XYT[1], new_T]


from math import *
import time
def goto(robot, xyt_position=None, translation_threshold=0.1, dryrun=False):
        """
        Moves the robot to the given goal state in
        the relative frame (base frame).
        :param xyt_position: The goal state of the form
                             (x,y,t) in the relative (base) frame.
        :type xyt_position: list
        """
        if xyt_position is None:
            xyt_position = [0.0, 0.0, 0.0]
        x = xyt_position[0]    # in meters
        y = xyt_position[1]    # in meters
        rot = xyt_position[2]  # in radians

        if sqrt(x * x + y * y) < translation_threshold:
            print("translation distance ", sqrt(x * x + y * y))
            print("rotate by ", rot)
            if not dryrun:
                robot.base.rotate_by(rot)
                robot.push_command()
                time.sleep(0.05)
            return True

        theta_1 = atan2(y, x)
        dist = sqrt(x ** 2 + y ** 2)

        if theta_1 > pi / 2:
            theta_1 = theta_1 - pi
            dist = -dist

        if theta_1 < -pi / 2:
            theta_1 = theta_1 + pi
            dist = -dist

        theta_2 = -theta_1 + rot

        # first rotate by theta1
        print("rotate by ", theta_1)
        if not dryrun:
            robot.base.rotate_by(theta_1, v_r=0.1)
            robot.push_command()
            time.sleep(0.2)
            while(abs(robot.base.left_wheel.status['vel']) >= 0.1 or abs(robot.base.right_wheel.status['vel']) >= 0.1):
                time.sleep(0.05)
        # move the distance
        print("translate by ", dist)
        if not dryrun:
            print("not a dryrun")
            robot.base.translate_by(dist, v_m=0.03)
            robot.push_command()
            time.sleep(5)
            # time.sleep(0.2)
            # while(abs(robot.base.left_wheel.status['vel']) >= 0.1 or abs(robot.base.right_wheel.status['vel']) >= 0.1):
            #     print(robot.base.left_wheel.status['vel'], robot.base.right_wheel.status['vel'])
            #     time.sleep(0.05)
        # second rotate by theta2
        print("rotate by ", theta_2)
        if not dryrun:
            robot.base.rotate_by(theta_2)
            robot.push_command()
            time.sleep(0.2)
            while(abs(robot.base.left_wheel.status['vel']) >= 0.1 or abs(robot.base.right_wheel.status['vel']) >= 0.1):
                time.sleep(0.05)
        return True
