import numpy as np
from scipy.spatial.transform import Rotation
import cv2
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

def is_obstacle_ahead(dist, depth_fn): 
    rgb, depth_map = depth_fn()
    depth_thresh = 5 # Threshold for SAFE distance (in m)

    # Mask to segment regions with depth less than threshold
    mask = cv2.inRange(depth_map,0,depth_thresh)
    
    obstacle_dist, _ = cv2.meanStdDev(depth_map, mask=mask)
    obstacle_dist = np.squeeze(obstacle_dist)
    print(f'init obstacle_dist {obstacle_dist}')
    
    # Check if a significantly large obstacle is present and filter out smaller noisy regions
    if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:
        
        image_gray = cv2.cvtColor(cv2.bitwise_and(rgb, rgb, mask=mask), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image_gray, 100, 200)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)

        # Contour detection 
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        
        thresh = 0.001*mask.shape[0]*mask.shape[1]
        mask2 = np.zeros_like(mask)
        
        minix = 0
        for i in range(len(cnts)):
            x = cnts[i]
            if cv2.contourArea(x) > thresh:
                mask3 = np.zeros_like(mask)
                # finding average depth of region represented by the largest contour 
                cv2.drawContours(mask3, cnts, i, (255), -1)
                cv2.drawContours(mask2, cnts, i, (255), -1)
                depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask3)
                depth_mean = np.squeeze(depth_mean)
                # pick the contour with the minimum depth average
                if depth_mean < obstacle_dist:
                    minix = i
                obstacle_dist = min(obstacle_dist, depth_mean)
        
        min_mask = np.zeros_like(mask)
        cv2.drawContours(min_mask, cnts, minix, (200), -1)
        
        depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)
        depth_mean = np.squeeze(depth_mean)
        obstacle_dist = min(obstacle_dist, depth_mean) * cos(radians(45))
        obstacle_dist -= 0.5 # buffer distance
        print(f'obstacle_dist {obstacle_dist}, dist {dist}')
        if obstacle_dist < dist:
            print(f'OBSTACLE!!')
    
    return obstacle_dist < dist


def goto(robot, xyt_position=None, translation_threshold=0.1, dryrun=False, depth_fn=None):
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
            print(f'rotate by {xyt_position[2], rot}')
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
        theta_1 = np.sign(theta_1) * (abs(theta_1) % radians(360))
        print("rotate by theta_1", theta_1)
        if not dryrun:
            robot.base.rotate_by(theta_1)
            robot.push_command()
            time.sleep(0.2)
            while(abs(robot.base.left_wheel.status['vel']) >= 0.1 or abs(robot.base.right_wheel.status['vel']) >= 0.1):
                time.sleep(0.05)
        # move the distance
        print("translate by ", dist)
        if not dryrun:
            print("not a dryrun")

            # # check here if obstacle is within dist, return false.
            # if is_obstacle_ahead(abs(dist), depth_fn):
            #     return False

            robot.base.translate_by(dist, v_m=0.1)
            robot.push_command()
            time.sleep(5)
            # TODO: switch to checking robot's status, instead of the sleep
            # time.sleep(0.2)
            # while(abs(robot.base.left_wheel.status['vel']) >= 0.1 or abs(robot.base.right_wheel.status['vel']) >= 0.1):
            #     print(robot.base.left_wheel.status['vel'], robot.base.right_wheel.status['vel'])
            #     time.sleep(0.05)
        # second rotate by theta2
        
        theta_2 = np.sign(theta_2) * (abs(theta_2) % radians(360))
        print("rotate by theta_2", theta_2)
        if not dryrun:
            robot.base.rotate_by(theta_2)
            robot.push_command()
            time.sleep(0.2)
            while(abs(robot.base.left_wheel.status['vel']) >= 0.1 or abs(robot.base.right_wheel.status['vel']) >= 0.1):
                time.sleep(0.05)
        return True
