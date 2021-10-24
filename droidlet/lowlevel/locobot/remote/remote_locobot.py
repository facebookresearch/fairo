"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# python -m Pyro4.naming -n <MYIP>
import Pyro4
from pyrobot import Robot
from pyrobot.locobot.camera import DepthImgProcessor
import numpy as np
from scipy.spatial.transform import Rotation
import logging
import os
import json
import skfmm
import skimage
from pyrobot.locobot.camera import DepthImgProcessor
from pyrobot.locobot.base_control_utils import LocalActionStatus
from slam_pkg.utils import depth_util as du

Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.ITER_STREAMING = True
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


@Pyro4.expose
class RemoteLocobot(object):
    """PyRobot interface for the Locobot.

    Args:
        backend (string): the backend for the Locobot ("habitat" for the locobot in Habitat, and "locobot" for the physical LocoBot)
        (default: locobot)
        backend_config (dict): the backend config used for connecting to Habitat (default: None)
    """

    def __init__(self, backend="locobot", backend_config=None, noisy=False):
        if backend == "habitat":
            if backend_config["physics_config"] == "DEFAULT":
                assets_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../tests/test_assets")
                )
                backend_config["physics_config"] = os.path.join(
                    assets_path, "default.phys_scene_config.json"
                )
            backend_config["noisy"] = noisy
            print("backend_config", backend_config)
            self.backend_config = backend_config
            # we do it this way to have the ability to restart from the client at arbitrary times
            self.restart_habitat()
        else:
            raise RuntimeError("Unknown backend", backend)

        # check skfmm, skimage in installed, its necessary for slam
        self._done = True
        self.backend = backend

    def restart_habitat(self):
        if hasattr(self, "_robot"):
            del self._robot
        backend_config = self.backend_config

        self._robot = Robot("habitat", common_config=backend_config)
        from habitat_utils import reconfigure_scene
        # adds objects to the scene, doing scene-specific configurations
        reconfigure_scene(self, backend_config["scene_path"])
        from pyrobot.locobot.camera import DepthImgProcessor

        if hasattr(self, "_dip"):
            del self._dip
        self._dip = DepthImgProcessor(cfg_filename="realsense_habitat.yaml")

    def test_connection(self):
        print("Connected!!")  # should print on server terminal
        return "Connected!"  # should print on client terminal

    def get_img_resolution(self):
        """return height and width"""
        if self.backend == "habitat":
            return (512, 512)
        else:
            return None

    def get_pcd_data(self):
        """Gets all the data to calculate the point cloud for a given rgb, depth frame."""
        rgb, depth = self._robot.camera.get_rgb_depth()
        depth *= 1000  # convert to mm
        # cap anything more than np.power(2,16)~ 65 meter
        depth[depth > np.power(2, 16) - 1] = np.power(2, 16) - 1
        depth = depth.astype(np.uint16)
        if self.backend == "habitat":
            cur_state = self._robot.camera.agent.get_state()
            cur_sensor_state = cur_state.sensor_states["rgb"]
            initial_rotation = cur_state.rotation
            rot_init_rotation = self._robot.camera._rot_matrix(initial_rotation)
            relative_position = cur_sensor_state.position - cur_state.position
            relative_position = rot_init_rotation.T @ relative_position
            cur_rotation = self._robot.camera._rot_matrix(cur_sensor_state.rotation)
            cur_rotation = rot_init_rotation.T @ cur_rotation
            return rgb, depth, cur_rotation, -relative_position
        return None

    def go_to_absolute(
        self,
        xyt_position,
        use_map=False,
        close_loop=False,
        smooth=False,
        wait=True,
    ):
        """Moves the robot base to given goal state in the world frame.

        :param xyt_position: The goal state of the form (x,y,yaw)
                             in the world (map) frame.
        :param use_map: When set to "True", ensures that controller is
                        using only free space on the map to move the robot.
        :param close_loop: When set to "True", ensures that controller
                           is operating in open loop by taking
                           account of odometry.
        :param smooth: When set to "True", ensures that the motion
                       leading to the goal is a smooth one.

        :type xyt_position: list or np.ndarray
        :type use_map: bool
        :type close_loop: bool
        :type smooth: bool
        """
        if self._done:
            self._done = False
            self._robot.base.go_to_absolute(
                xyt_position, use_map=use_map, close_loop=close_loop, smooth=smooth, wait=wait)
            self._done = True

    def go_to_relative(
        self,
        xyt_position,
        use_map=False,
        close_loop=False,
        smooth=False,
        wait=True,
    ):
        """Moves the robot base to the given goal state relative to its current
        pose.

        :param xyt_position: The  relative goal state of the form (x,y,yaw)
        :param use_map: When set to "True", ensures that controller is
                        using only free space on the map to move the robot.
        :param close_loop: When set to "True", ensures that controller is
                           operating in open loop by taking
                           account of odometry.
        :param smooth: When set to "True", ensures that the
                       motion leading to the goal is a smooth one.

        :type xyt_position: list or np.ndarray
        :type use_map: bool
        :type close_loop: bool
        :type smooth: bool
        """
        if self._done:
            self._done = False
            self._robot.base.go_to_relative(
                xyt_position, use_map=use_map, close_loop=close_loop, smooth=smooth, wait=wait
                )
            self._done = True

    @Pyro4.oneway
    def stop(self):
        """stops robot base movement."""
        self._robot.base.stop()

    def get_base_state(self, state_type="odom"):
        """Returns the  base pose of the robot in the (x,y, yaw) format as
        computed either from Wheel encoder readings or Visual-SLAM.

        :param state_type: Requested state type. Ex: Odom, SLAM, etc

        :type state_type: string

        :return: pose of the form [x, y, yaw]
        :rtype: list
        """
        return self._robot.base.get_state(state_type)

    # Common wrapper
    def command_finished(self):
        """Returns whether previous executed command finished execution.

        return: command execution state [True for finished execution, False for still executing]
        rtype: bool
        """
        return self._done

    # Camera wrapper
    def get_depth(self):
        """Returns the depth image perceived by the camera.

        :return: depth image in meters, dtype-> float32
        :rtype: np.ndarray or None
        """
        depth = self._robot.camera.get_depth()
        if depth is not None:
            return depth
        return None

    def get_depth_bytes(self):
        """Returns the depth image perceived by the camera.

        :return: depth image in meters, dtype-> bytes
        :rtype: np.ndarray or None
        """
        depth = self._robot.camera.get_depth().astype(np.int64)
        if depth is not None:
            return depth.tobytes()
        return None

    def get_intrinsics(self):
        """Returns the intrinsic matrix of the camera.

        :return: the intrinsic matrix (shape: :math:`[3, 3]`)
        :rtype: list
        """
        intrinsics = self._robot.camera.get_intrinsics()
        if intrinsics is not None:
            return intrinsics.tolist()
        return None

    def get_rgb(self):
        """Returns the RGB image perceived by the camera.

        :return: image in the RGB, [h,w,c] format, dtype->uint8
        :rtype: np.ndarray or None
        """
        rgb = self._robot.camera.get_rgb()
        if rgb is not None:
            return rgb
        return None

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
            pts = du.transform_pose(pts, self._robot.base.get_state("odom"))
        return pts, colors

    def pix_to_3dpt(self, rs, cs, in_cam=False):
        """Get the 3D points of the pixels in RGB images in metric unit.

        :param rs: rows of interest in the RGB image.
                   It can be a list or 1D numpy array
                   which contains the row indices.
        :param cs: columns of interest in the RGB image.
                   It can be a list or 1D numpy array
                   which contains the column indices.
        :param in_cam: return points in camera frame,
                       otherwise, return points in base frame

        :type rs: list or np.ndarray
        :type cs: list or np.ndarray
        :type in_cam: bool

        :returns: tuple (pts, colors)

                  pts: point coordinates in metric unit
                  (shape: :math:`[N, 3]`)

                  colors: rgb values
                  (shape: :math:`[N, 3]`)

        :rtype: tuple(list, list)
        """
        pts, colors = self._robot.camera.pix_to_3dpt(rs, cs, in_cam=in_cam)
        return pts.tolist(), colors.tolist()

    def dip_pix_to_3dpt(self):
        """Return the point cloud at current time step in robot base frame.

        :returns: point coordinates (shape: :math:`[N, 3]`) in metric unit
        :rtype: list
        """
        logging.info("dip_pix_to_3dpt")
        depth = self._robot.camera.get_depth()
        h = depth.shape[0]
        w = depth.shape[1]

        xs = np.repeat(np.arange(h), w).ravel()
        ys = np.repeat(np.arange(w)[None, :], h, axis=0).ravel()

        pts = self.pix_to_3dpt(xs, ys)
        pts = np.around(pts, decimals=2)
        logging.info("exiting pix_to_3dpt")
        return pts

    def get_transform(self, src_frame, dst_frame):
        """Return the transform from the src_frame to dest_frame.

        :param src_frame: source frame
        :param dst_frame: destination frame

        :type src_frame: str
        :type dst_frame: str

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

    def get_base_status(self):
        status = self._robot.base._as.get_state()
        if status == LocalActionStatus.ACTIVE:
            return "ACTIVE"
        elif status == LocalActionStatus.SUCCEEDED:
            return "SUCCEEDED"
        elif status == LocalActionStatus.PREEMPTED:
            return "PREEMPTED"
        else:
            return "UNKNOWN"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pass in server device IP")
    parser.add_argument(
        "--ip",
        help="Server device (robot) IP. Default is 192.168.0.0",
        type=str,
        default="192.168.0.0",
    )
    parser.add_argument(
        "--backend",
        help="PyRobot backend to use (locobot | habitat). Default is locobot",
        type=str,
        default="locobot",
    )
    parser.add_argument(
        "--backend_config",
        help="Optional config argument to be passed to the backend."
        "Currently mainly used to pass Habitat environment path",
        type=json.loads,
        default='{"scene_path": "/Replica-Dataset/apartment_0/habitat/mesh_semantic.ply", \
            "physics_config": "DEFAULT"}',
    )
    parser.add_argument(
         "--noisy",
        type=bool,
        default=os.getenv("NOISY_HABITAT", "False").lower() in ("true", "True"),
        help="Set to True to load habitat with rgb, depth and movement noise models"
    )

    args = parser.parse_args()

    np.random.seed(123)

    if args.backend == "habitat":
        # GLContexts in general are thread local
        # The PyRobot <-> Habitat integration is not thread-aware / thread-configurable,
        # so our only option is to disable Pyro4's threading, and instead switch to
        # multiplexing (which isn't too bad)
        Pyro4.config.SERVERTYPE = "multiplex"

    with Pyro4.Daemon(args.ip) as daemon:
        robot = RemoteLocobot(
            backend=args.backend, 
            backend_config=args.backend_config,
            noisy=args.noisy,
        )
        robot_uri = daemon.register(robot)
        with Pyro4.locateNS() as ns:
            ns.register("remotelocobot", robot_uri)

        print("Server is started...")
        daemon.requestLoop()


# Below is client code to run in a separate Python shell...
# import Pyro4
# robot = Pyro4.Proxy("PYRONAME:remotelocobot")
# robot.go_home()
