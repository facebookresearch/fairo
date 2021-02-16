"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

# python -m Pyro4.naming -n <MYIP>

import Pyro4
import numpy as np
import pickle
import time
#from scipy.spatial.transform import Rotation
import logging
import os
import json
import sys
import uuid
import pyrealsense2 as rs
import robomaster
from robomaster import robot
#import skfmm
#import skimage
#from slam.slam import Slam

Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.ITER_STREAMING = True

CAMERA_HEIGHT = .3
CH = 480
CW = 640
FREQ = 10

class StoreCallback:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
    def __call__(self, info):
        self.storage[self.name] = info

@Pyro4.expose
class RemoteDJI(object):
    """on-jetson interface for the DJI-robomaster

    Args:
    """
    def __init__(self):
        self.callback_storage = {}
        self.dji = None
        self.connect_to_dji()
        self.connect_to_realsense()
        #for now don't bother clearing this...
        self.command_objects = {}
        
        # self.decimate_filter = rs.decimation_filter()
        # self.decimate_filter.set_option(rs.option.filter_magnitude, 2)

        # check skfmm, skimage in installed, its necessary for slam
#               self._slam = Slam(self._robot, backend)
#        self._slam.set_goal(
#            (19, 19, 0)
#        )  # set  far away goal for exploration, default map size [-20,20]
#        self._slam_step_size = 25  # step size in cm
#        self._done = True


    def connect_to_dji(self):
        self.dji = robot.Robot()
        self.dji.initialize(conn_type="rndis")
        
        cs = self.callback_storage
        C = StoreCallback
        self.dji.chassis.sub_position(freq=FREQ, callback=C(cs, "position"))
        self.dji.chassis.sub_attitude(freq=FREQ, callback=C(cs, "chassis_pitchyaw"))
        self.dji.chassis.sub_status(freq=FREQ, callback=C(cs, "bot_flags"))
        self.dji.gripper.sub_status(freq=FREQ, callback=C(cs, "gripper_status"))
        self.dji.robotic_arm.sub_position(freq=FREQ, callback=C(cs, "arm_position"))
        self.dji.servo.sub_servo_info(freq=FREQ, callback=C(cs, "servo"))
        #FIXME
        print("connected to dji")
        self.dji.led.set_led(r=np.random.randint(0,250),
                             g=np.random.randint(0,250),
                             b=np.random.randint(0,250))
        # TODO armor, battery, microphone

    def disconnect_dji(self):
        if self.dji is not None:
            self.dji.close()

    def retry_dji(self, sleep=0.0):
        self.disconnect_dji()
        if sleep > 0:
            time.sleep(sleep)
        self.connect_to_dji()
        

    def get_all_dji_info(self):
        return self.callback_storage

    def connect_to_realsense(self):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, CW, CH, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, CW, CH, rs.format.z16, 30)
        pipeline = rs.pipeline()
        pipeline.start(cfg)
        self.realsense = pipeline
        profile = pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        i = depth_profile.get_intrinsics()
        self.intrinsic_mat = np.array([[i.fx, 0,    i.ppx],
                                       [0,    i.fy, i.ppy],
                                       [0,    0,    1]])
        self.depth_img_size = [i.height, i.width]
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        #FIXME (build a test connection method        
        print("connected to realsense")

    def get_intrinsics(self):
        return self.intrinsic_mat
        
    def test_connection(self):
        print("Connected!!")  # should print on server terminal
        return "Connected!"  # should print on client terminal

    def record_action(self, action):
        action_id = uuid.uuid4().hex
        self.command_objects[action_id] = action
        return action_id

    def get_action_state(self, uid):
        s = self.command_objects.get(uid)
        if s is not None:
            return s.state
        else:
            return None

    # TODO FIXME!!!!!
    # 0: use .drive_speed() instead of .move()
    #   to control; right now no easy stop inside one of the moves
    # 1: use map + vslam
    # 2: stop if something is too close    
    def go_to_relative(self,
                       x=0,
                       y=0,
                       yaw=0,
                       xy_speed=1.0,
                       yaw_speed=40.0,
                       use_map=False,
                       close_loop=True,
                       smooth=False):
        """Moves the robot base to the given goal state relative to its current
        pose.

        :param xyt_position: The  relative goal state of the form (x,y,yaw)
        :param use_map: When set to "True", ensures that controller is
                        using only free space on the map to move the robot. NOT IMPLEMENTED!!
        :param close_loop: When set to "True", ensures that controller is
                           operating in open loop by taking
                           account of odometry.  NOT IMPLEMENTED!!
        :param smooth: When set to "True", ensures that the
                       motion leading to the goal is a smooth one.  NOT IMPLEMENTED!!

        :type xyt_position: list or np.ndarray
        :type use_map: bool
        :type close_loop: bool
        :type smooth: bool
        """
        s = self.dji.chassis.move(x=x,
                                  y=y,
                                  z=yaw,
                                  xy_speed=xy_speed,
                                  z_speed=yaw_speed)
        uid = self.record_action(s)
        return uid


    #TODO different dji have different servo numbering.....
    def set_servo(self, index, angle):
        """ set the servo "index" to angle "angle". """
        s = self.dji.servo.moveto(index=index, angle=angle)
        uid = self.record_action(s)
        return uid

    # FIXME!!! not implemented
#    @Pyro4.oneway
#    def stop(self):
#        """stops robot base movement."""
#        self._robot.base.stop()


    #FIXME add SLAM, add a re-zero method
    def get_base_state(self, state_type):
        """Returns the  base pose of the robot in the (x,y, yaw) format as
        computed either from Wheel encoder readings or Visual-SLAM.

        :param state_type: Requested state type. Ex: Odom, SLAM, etc  !!FIXME, only Odom supported

        :type state_type: string

        :return: pose of the form [x, y, yaw]
        :rtype: list
        """
        return self.callback_storage["position"]



    def use_gripper(self, open_or_close):
        """Commands gripper to open or close fully."""
        if open_or_close == "open":
            s = self.dji.gripper.open(power=20)
        elif open_or_close == "close":        
            s = self.dji.gripper.close(power=20)
        else:
            return
        uid = self.record_action(s)
        return uid

    def get_gripper_state(self):
        """Return the gripper state.

        """
        return self.callback_storage["gripper"]


    def move_arm_to(self, x=0, y=0):
        """moves arm to x, y position"""
        s = self.dji.robotic_arm.moveto(x=x, y=y)
        uid = self.record_action(s)
        return uid

    def pitch_to_xy(self, pitch):
        r= 11/2
        if pitch >= 22 + r:
            return (125, 150)
        elif pitch <= 22 + r  and pitch > 22 - r:
            return (105, 135)
        elif pitch <= 11 + r and pitch > 11 - r:
            return (95, 105)
        elif pitch <= r and pitch > -r:
            return (88, 80)
        elif pitch <= -11 + r and pitch > -11 - r:
            return (110, 60)
        elif pitch <= -22 + r and pitch > -22 - r:
            return (110, 30)
        else:
            return (70,30)

    def set_pitch(self, pitch):
        x, y = self.pitch_to_xy(pitch)
        return self.move_arm_to(x=x, y=y)


    def recenter_arm(self):
        """run dji arm recenter"""
        s = self.dji.robotic_arm.recenter()
        uid = self.record_action(s)
        return uid
    
    def get_joint_positions(self):
        """Return arm joint angles order-> base_join index 0, wrist joint index
        -1.

        :return: joint_angles in radians
        :rtype: list
        """
        return self._robot.arm.get_joint_angles().tolist()

    def get_joint_velocities(self):
        """Return the joint velocity order-> base_join index 0, wrist joint
        index -1.

        :return: joint_angles in rad/sec
        :rtype: list
        """
        return self._robot.arm.get_joint_velocities().tolist()


    # Camera wrapper

    def get_img_resolution(self):
        """ return height and width"""
        return self.depth_img_size

    def get_rgb_depth(self):
        frames = None
        while not frames:
            frames = self.realsense.wait_for_frames()
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.single)/1000.0
            color_image = np.asanyarray(color_frame.get_data())
            
        return color_image, depth_image

    def get_pcd_data(self):
        """Gets all the data to calculate the point cloud for a given rgb, depth frame."""
        rgb, depth = self.get_rgb_depth()
        depth *= 1000  # convert to mm
        # cap anything more than np.power(2,16)~ 65 meter
        depth[depth > np.power(2, 16) - 1] = np.power(2, 16) - 1
        depth = depth.astype(np.uint16)
        #FIXME THIS IS BROKEN!! (deal with pitch)
        trans = [0, 0, CAMERA_HEIGHT]
        rot = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1.,0.]])
        T = np.eye(4)
        T[:3,:3] = rot
        T[:3,3] = trans 
#        trans, rot, T = self.realsense.get_link_transform(
#            self._robot.camera.cam_cf, self._robot.camera.base_f
#        )
        base2cam_trans = np.array(trans).reshape(-1, 1)
        base2cam_rot = np.array(rot)
        return rgb, depth, base2cam_rot, base2cam_trans


    def get_depth(self):
        """Returns the depth image perceived by the camera.

        :return: depth image in meters, dtype-> float32
        :rtype: np.ndarray or None
        """
        frames = self.realsense.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        try:
            depth_frame = np.asanyarray(depth_frame.get_data()).astype(np.single)/1000
        except:
            return None
        return depth_frame

    def get_depth_bytes(self):
        """Returns the depth image perceived by the camera.

        :return: depth image in meters, dtype-> bytes
        :rtype: np.ndarray or None
        """
        depth = self.get_depth()
        if depth is not None:
            return depth.astype(np.int64).tobytes()
        return None

#    def get_intrinsics(self):
#        """Returns the intrinsic matrix of the camera.
#
#        :return: the intrinsic matrix (shape: :math:`[3, 3]`)
#        :rtype: list
#        """
#        intrinsics = self._robot.camera.get_intrinsics()
#        if intrinsics is not None:
#            return intrinsics.tolist()
#        return None

    def get_rgb(self):
        """Returns the RGB image perceived by the camera.

        :return: image in the RGB, [h,w,c] format, dtype->uint8
        :rtype: np.ndarray or None
        """
        frames = self.realsense.wait_for_frames()
        rgb_frame = frames.get_color_frame()
        try:
            rgb_frame = np.asanyarray(rgb_frame.get_data())
        except:
            return None
        return rgb_frame

    def get_rgb_bytes(self):
        """Returns the RGB image perceived by the camera.

        :return: image in the RGB, [h,w,c] format, dtype->bytes
        :rtype: np.ndarray or None
        """
        rgb = self.get_rgb()
        if rgb is not None:
            return rgb.astype(np.int64).tobytes()
        return None

    def transform_pose(self, XYZ, current_pose):
        """
        Transforms the point cloud into geocentric frame to account for
        camera position

        Args:
            XYZ                     : ...x3
            current_pose            : camera position (x, y, theta (radians))
        Returns:
            XYZ : ...x3
        """
        R = Rotation.from_euler("Z", current_pose[2]).as_matrix()
        XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape((-1, 3))
        XYZ[:, 0] = XYZ[:, 0] + current_pose[0]
        XYZ[:, 1] = XYZ[:, 1] + current_pose[1]
        return XYZ

    def get_current_pcd(self, in_cam=False, in_global=False):
        """Return the point cloud at current time step.

        :param in_cam: return points in camera frame,
                       otherwise, return points in base frame

        :type in_cam: bool

        :returns: tuple (pts, colors)

                  pts: point coordinates (shape: :math:`[N, 3]`) in metric unit

                  colors: rgb values (shape: :math:`[N, 3]`)
        :rtype: tuple(list, list)
        """
        pts, colors = self._robot.camera.get_current_pcd(in_cam=in_cam)

        if in_global:
            pts = self.transform_pose(pts, self._robot.base.get_state("odom"))
        return pts, colors


    def get_transform(self, src_frame, dst_frame):
        """Return the transform from the src_frame to dest_frame.

        :param src_frame: source frame
        :param dest_frame: destination frame

        :type src_frame: str
        :type dest_frame: str

        :return:tuple (trans, rot_mat, quat )
                trans: translational vector (shape: :math:`[3, 1]`)
                rot_mat: rotational matrix (shape: :math:`[3, 3]`)
                quat: rotational matrix in the form of quaternion (shape: :math:`[4,]`)

        :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
        """
        return self._robot.arm.get_transform(src_frame, dst_frame)

    # Camera pan wrapper
    def get_pan(self):
        """Return the current pan joint angle of the robot camera.

        :return:Pan joint angle in radian
        :rtype: float
        """
        return self._robot.camera.get_pan()

    def get_camera_state(self):
        """Return the current pan and tilt joint angles of the robot camera in
        radian.

        :return: A list the form [pan angle, tilt angle]
        :rtype: list
        """
        return self._robot.camera.get_state()

    def get_tilt(self):
        """Return the current tilt joint angle of the robot camera in radian.

        :return:tilt joint angle
        :rtype: float
        """
        return self._robot.camera.get_tilt()

    @Pyro4.oneway
    def reset(self):
        """This function resets the pan and tilt joints of robot camera by
        actuating them to their home configuration [0.0, 0.0]."""
        if self._done:
            self._done = False
            self._robot.camera.reset()
            self._done = True
    @Pyro4.oneway
    def set_pan(self, pan, wait=True):
        """Sets the pan joint angle of robot camera to the specified value.

        :param pan: value to be set for pan joint in radian
        :param wait: wait until the pan angle is set to
                     the target angle.

        :type pan: float
        :type wait: bool
        """
        if self._done:
            self._done = False
            self._robot.camera.set_pan(pan, wait=wait)
            self._done = True

    @Pyro4.oneway
    def set_pan_tilt(self, pan, tilt, wait=True):
        """Sets both the pan and tilt joint angles of the robot camera  to the
        specified values.

        :param pan: value to be set for pan joint in radian
        :param tilt: value to be set for the tilt joint in radian
        :param wait: wait until the pan and tilt angles are set to
                     the target angles.

        :type pan: float
        :type tilt: float
        :type wait: bool
        """
        if self._done:
            self._done = False
            self._robot.camera.set_pan_tilt(pan, tilt, wait=wait)
            self._done = True

    @Pyro4.oneway
    def set_tilt(self, tilt, wait=True):
        """Sets the tilt joint angle of robot camera to the specified value.

        :param tilt: value to be set for the tilt joint in radian
        :param wait: wait until the tilt angle is set to
                     the target angle.

        :type tilt: float
        :type wait: bool
        """
        if self._done:
            self._done = False
            self._robot.camera.set_tilt(tilt, wait=wait)
            self._done = True

    # grasping wrapper
    def grasp(self, dims=[(240, 480), (100, 540)]):
        """
        :param dims: List of tuples of min and max indices of the image axis to be considered for grasp search
        :type dims: list
        :return:
        """
        if self._done:
            self._done = False
            # TODO: in reset, pan of camera is set to point to ground, may not need that part
            # success = self._grasper.reset()
            if not success:
                return False
            # grasp_pose = self._grasper.compute_grasp(dims=dims)
            # self._grasper.grasp(grasp_pose)
            self._done = True
            return True

    # slam wrapper
    def explore(self):
        if self._done:
            self._done = False
            self._slam.take_step(self._slam_step_size)
            self._done = True
            return True


if __name__ == "__main__":
#    R = RemoteDJI()
#    cow = """
    
    import argparse

    parser = argparse.ArgumentParser(description="Pass in server device IP")
    parser.add_argument(
        "--name",
        help="name of robot for pyro",
        type=str,
        default="dji",
    )
    parser.add_argument(
        "--host",
        help="this ip",
        type=str,
        default="",
    )
    args = parser.parse_args()

    np.random.seed(123)
    with Pyro4.Daemon(host=args.host) as daemon:
        robot_uri = daemon.register(RemoteDJI)
        with Pyro4.locateNS() as ns:
            ns.register(args.name, robot_uri)
        print("dji uri is " + str(robot_uri))
        daemon.requestLoop()
        print("Server is started...")

# Below is client code to run in a separate Python shell...
# import Pyro4
# robot = Pyro4.Proxy("PYRONAME:remotelocobot")
# robot.go_home()
