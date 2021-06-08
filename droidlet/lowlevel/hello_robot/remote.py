"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# python -m Pyro4.naming -n <MYIP>
import Pyro4
from stretch_body.robot import Robot
import stretch_body.pimu as pimu
from colorama import Fore, Back, Style
import stretch_body.hello_utils as hu
hu.print_stretch_re_use()
import numpy as np
import logging
import os
import json
import pyrealsense2 as rs
import cv2

# Configure depth and color streams
CAMERA_HEIGHT = .3
CH = 480
CW = 640
FREQ = 30

# fps=30
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, CW, CH, rs.format.z16, fps)
# config.enable_stream(rs.stream.color, CW, CH, rs.format.bgr8, fps)
# pipeline.start(config)


Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.ITER_STREAMING = True

def val_in_range(val_name, val,vmin, vmax):
    p=val <=vmax and val>=vmin
    if p:
        print(Fore.GREEN +'[Pass] ' + val_name + ' with ' + str(val))
    else:
        print(Fore.RED +'[Fail] ' + val_name + ' with ' +str(val)+ ' out of range ' +str(vmin) + ' to ' + str(vmax))

# #####################################################


@Pyro4.expose
class RemoteLocobot(object):
    """PyRobot interface for the Locobot.

    Args:
        backend (string): the backend for the Locobot ("habitat" for the locobot in Habitat, and "locobot" for the physical LocoBot)
        (default: locobot)
        backend_config (dict): the backend config used for connecting to Habitat (default: None)
    """

    def __init__(self):
        self._robot = Robot()
        self._robot.startup()
        if not self._robot.is_calibrated():
            self._robot.home() #blocking
        # self._check_battery()
        self._connect_to_realsense()
    
    def _connect_to_realsense(self):
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
        print("connected to realsense")
        
    def _check_battery(self):
        p=pimu.Pimu()
        if not p.startup():
            exit()
        p.pull_status()
        val_in_range('Voltage',p.status['voltage'], vmin=p.config['low_voltage_alert'], vmax=14.0)
        val_in_range('Current',p.status['current'], vmin=0.1, vmax=p.config['high_current_alert'])
        val_in_range('CPU Temp',p.status['cpu_temp'], vmin=15, vmax=80)
        print(Style.RESET_ALL)
        p.stop()

    def get_intrinsics(self):
        return self.intrinsic_mat.tolist()
    
    def get_img_resolution(self):
        return (CH, CW)
    
    def get_status(self):
        return self._robot.get_status()
    
    def get_base_state(self):
        s = self._robot.get_status()
        return (s['base']['x'], s['base']['y'], s['base']['theta'])
    
    def get_pan(self):
        s = self._robot.get_status()
        return s['head']['head_pan']['pos']

    def get_tilt(self):
        s = self._robot.get_status()
        return s['head']['head_tilt']['pos']
    
    def test_connection(self):
        print("Connected!!")  # should print on server terminal
        # print(self._robot.get_status())
        return "Connected!"  # should print on client terminal
    
    def home(self):
        self._robot.home()
    
    def stow(self):
        self._robot.stow()

    def translate_by(self, x_m):
        self._robot.base.translate_by(x_m)
        self._robot.push_command()

    def rotate_by(self, x_r):
        self._robot.base.rotate_by(x_r)
        self._robot.push_command()
    
    def get_rgb_depth(self):
        frames = None
        while not frames:
            # frames = self.realsense.wait_for_frames()
            # depth_frame = frames.get_depth_frame()
            # color_frame = frames.get_color_frame()
            # print(type(color_frame))
            
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

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=-0.04, beta=255.0), cv2.COLORMAP_OCEAN)
            color_image = np.moveaxis(color_image, 0, 1)
            depth_colormap = np.moveaxis(depth_colormap, 0, 1)

            # return color_frame, depth_frame
        print(type(color_image))
        # return pickle.dumps(color_image, protocol=2), pickle.dumps(depth_image, protocol=2)
        return color_image.tolist(), depth_image.tolist()

    def get_pcd_data(self):
        """Gets all the data to calculate the point cloud for a given rgb, depth frame."""
        rgb, depth = self.get_rgb_depth()
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
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
        print('shapes ...{}, {}, {}, {}'.format(rgb.shape, depth.shape, base2cam_rot.shape, base2cam_trans.shape))
        return rgb.tolist(), depth.tolist(), base2cam_rot.tolist(), base2cam_trans.tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pass in server device IP")
    parser.add_argument(
        "--ip",
        help="Server device (robot) IP. Default is 192.168.0.0",
        type=str,
        default="172.20.7.104",
    )
    
    args = parser.parse_args()

    np.random.seed(123)

    with Pyro4.Daemon(args.ip) as daemon:
        robot = RemoteLocobot()
        robot_uri = daemon.register(robot)
        with Pyro4.locateNS() as ns:
            ns.register("remotelocobot", robot_uri)

        print("Server is started...")
        daemon.requestLoop()


# Below is client code to run in a separate Python shell...
# import Pyro4
# robot = Pyro4.Proxy("PYRONAME:remotelocobot")
# robot.go_home()
