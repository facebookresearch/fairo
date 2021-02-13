"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import numpy as np
import cv2
import os
import math
import copy
import robomaster
from robomaster import robot
import Pyro4
from Pyro4 import naming
from scipy.spatial.transform import Rotation
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")


#from perception import RGBDepth
#from objects import Marker, Pos


#FIXME this is exact copy from locobot utils, don't copy
def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
    current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = Rotation.from_euler("Z", current_pose[2]).as_matrix()
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape((-1, 3))
    XYZ[:, 0] = XYZ[:, 0] + current_pose[0]
    XYZ[:, 1] = XYZ[:, 1] + current_pose[1]
    return XYZ


def to_interpreter_xzy(xyyaw):
    """ converts dji's (x, y, yaw) 
        with forward = (1, 0, 0)
        and strafe right = (0, -1, 0)
        to interpreter's (x, z, yaw)
        with forward = (0, 1, 0)
        and strafe right = (1, 0, 0) """
    return xyyaw[1], xyyaw[0], xyyaw[2]

def from_interpreter_xzy(xzy):
    """ converts interpreter's (x, z, yaw)
        with forward = (0, 1, 0)
        and strafe right = (1, 0, 0)
        to dji's (x, y, yaw) 
        with forward = (1, 0, 0)
        and strafe right = (0, -1, 0)
    """
    return xzy[1], xzy[0], xzy[2]


class Command:
    def __init__(self, command):
        self.command = command
    def run_command(self):
        return self.command()

#def pitch_to_xy(pitch, inner_angle):
#    """
#        |      x  
#        |A1*cosv M*cosu  
#        |   ||     ||
#    -------------:---:---------------
#        |     v /\ u           |
#        |      /ia\            |
#        | A1  /    \  A2 (mm)  | A2*sinu
#    A1  |    /      \          |
#    *   |   /        o x,y     |
#    sinv|  /                   -
#        | /                    | y
#
#    ia    = inner_angle, 
#    pitch = u + CAMERA_ANGLE (that is, CAMERA_ANGLE is the angle of the
#                   camera on the arm
#    u is the angle of arm2 from level (where level is 0) 
#    ARM1 is the length of arm 1 in mm
#    ARM2 is the length of arm 2 in mm
#    """    
#    u = pitch - CAMERA_ANGLE
#    v = 180 - inner_angle - u
#    u = u * np.pi / 180
#    v = v * np.pi / 180
#    x = ARM1 * np.cos(v) + ARM2*np.cos(u)
#    y = ARM1 * np.sin(v) - ARM2*np.sin(u)
#    return x, y



class RoboMasterMover:
    """Implements methods that call the physical interfaces of the RoboMaster.

    Arguments:
        robot_sn (string): the sn of the robomaster, if connected in networking mode
             if None, picks first one found

    """
    def __init__(self):
        nameserver = Pyro4.locateNS()
        jetson_uri = nameserver.lookup("dji")
        self.robot = Pyro4.Proxy(jetson_uri)

        intrinsic_mat = self.robot.get_intrinsics()
        intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
        img_resolution = self.robot.get_img_resolution()
        img_pixs = np.mgrid[0 : img_resolution[0] : 1, 0 : img_resolution[1] : 1]
        img_pixs = img_pixs.reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
        self.uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)

         
        self.current_actions = []
        self.queued_commands = []
        self.pitchyaw = []
        self.position = [0,0,0]
        self.bot_flags = []
        
        


    # TODO/FIXME!  instead of just True/False, return diagnostic messages
    # so e.g. if a grip attempt fails, the task is finished, but the status is a failure
    # TODO/FIXME! return an error when the robot freezes.  restart?
    def bot_step(self):
        try:
            unfinished = []
            finished = True
            for action_uid in self.current_actions:
                state = self.robot.get_action_state(action_uid)
                # TODO handle action_failed
                if state is not None and state != "action_succeeded":
                    finished = False
                    unfinished.append(s)
                    
            if not unfinished:
                if self.queued_commands:
                    finished = False
                    action_id = self.queued_commands[0].run_command()
                    unfinished.append(action_id)
                    del self.queued_commands[0]
            self.current_actions = unfinished
        except:
            # do better here?
            finished = True
#        print([s.state for s in self.current_actions])
        return finished

    #DJI camera is on arm, arm does not turn
    def get_pan(self):
        """get pan in degrees."""
        return 0.0

    # FIXME, dji tilt needs to be computed from gripper x,y 
    def get_tilt(self):
        """get pitch in radians."""
        return 0.0


    def move_absolute(self, xzyaw_positions):
        #FIXME use vslam! don't just go straight there
        #FIXME use .drive_speed() (with above) instead of .move()
        #   to control; right now no easy stop inside one of the moves
        for xzy in xzyaw_positions:
            x, y, yaw = from_interpreter_xzy(xzy)
            yaw = 360*yaw/(2*math.pi)
            def c():
                cx, cy, cyaw = self.get_base_pos_dji()
                dx, dy, dyaw = x-cx, y-cy, yaw-cyaw
                action_id = self.robot.go_to_relative(x=dx, y=dy, yaw=dyaw, xy_speed=1.0)
                return s
            self.queued_commands.append(Command(c))
    

    # FIXME this is BROKEN
    def look_at(self, obj_pos, yaw_deg, pitch_deg):
        """Executes "look at" by setting the pan, tilt of the camera or turning the base if required.

        Uses both the base state and object coordinates in canonical world coordinates to calculate
        expected yaw and pitch.

        Args:
            obj_pos (list): object coordinates as saved in memory.
            yaw_deg (float): yaw in degrees
            pitch_deg (float): pitch in degrees

        Returns:
            string "finished"
        """
        pan_rad, tilt_rad = 0.0, 0.0
        old_pan = self.position[2]
#        old_tilt = self.get_tilt()
        pos = self.get_base_pos()
        logging.debug(f"Current Locobot state (x, z, yaw): {pos}")
        if yaw_deg:
            pan = old_pan - float(yaw_deg)
        if pitch_deg:
            tilt = 0.0 #FIXME, dji tilt needs to be computed from gripper x,y 
#            tilt_rad = old_tilt - float(pitch_deg) * np.pi / 180
        if obj_pos is not None:
            logging.info(f"looking at x,y,z: {obj_pos}")
            pan_rad, tilt_rad = get_camera_angles([pos[0], CAMERA_HEIGHT, pos[1]], obj_pos)
            pan = pan_rad * 360 / (2 * np.pi)
            tilt = 0.0 #FIXME, dji tilt needs to be computed from gripper x,y 
            logging.debug(f"Returned new pan and tilt angles (radians): ({pan}, {tilt})")

        def c():
            return self.robot.go_to_relative(x=0.0, y=0.0, z=pan)
        self.queued_commands.append(Command(c))
        return "finished"

    #FIXME this is bROKEN
    def point_at(self, target_pos):
        pos = self.get_base_pos()
        yaw_rad, pitch_rad = get_camera_angles([pos[0], ARM_HEIGHT, pos[2]], target_pos)
        #FIXME, dji tilt needs to be computed from gripper x,y 
        pan = old_pan - yaw_rad*360/(2*pi)
        def c():
            return self.robot.go_to_relative(x=0.0, y=0.0, z=pan)
        self.queued_commands.append(Command(c))

    def get_base_pos_dji(self):
        """Return (x, z, yaw) in the interpreter absolute coordinates as
        
         yaw in degrees !
        """
        return self.position
        
    def get_base_pos(self):
        """Return (x, z, yaw) in the interpreter absolute coordinates as
        
         yaw in degrees !
        """        
        return to_interpreter_xzy(self.position)


    def get_rgb_depth(self):
        """Fetches rgb, depth and pointcloud in pyrobot world coordinates.

        Returns:
            an RGBDepth object
        """
        rgb, depth, rot, trans = self.robot.get_pcd_data()
        depth = depth.astype(np.float32)
        d = copy.deepcopy(depth)
        depth /= 1000.0
        depth = depth.reshape(-1)
        pts_in_cam = np.multiply(self.uv_one_in_cam, depth)
        pts_in_cam = np.concatenate((pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0)
        pts = pts_in_cam[:3, :].T
        pts = np.dot(pts, rot.T)
        pts = pts + trans.reshape(-1)
        pts = transform_pose(pts, self.robot.get_base_state("odom"))
        logging.info("Fetched all camera sensor input.")
        return RGBDepth(rgb, d, pts)

    def dance(self):
        self.robot.dance()

    def turn(self, yaw):
        """turns the bot by the yaw specified.

        Args:
            yaw: the yaw to execute in degree.
        """
        def c():
            return self.robot.go_to_relative(x=0.0, y=0.0, z=yaw)
        self.queued_commands.append(Command(c))

#######################
# TODO ? gripper stuff?
#######################
