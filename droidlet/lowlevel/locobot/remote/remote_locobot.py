"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
# python -m Pyro4.naming -n <MYIP>
import copy
import Pyro4
from pyrobot import Robot
import numpy as np
from scipy.spatial.transform import Rotation
import logging
import quaternion
import habitat_sim
import os
import open3d as o3d
from pyrobot.habitat.base_control_utils import LocalActionStatus
from slam_pkg.utils import depth_util as du
from obstacle_utils import is_obstacle
from droidlet.lowlevel.robot_mover_utils import (
    transform_pose,
)
from droidlet.dashboard.o3dviz import serialize as o3d_pickle

Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.ITER_STREAMING = True
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


@Pyro4.expose
class RemoteLocobot(object):
    """PyRobot interface for the Locobot.

    Args:
        scene_path (str): the path to the scene file to load in habitat
    """

    def __init__(self, scene_path, noisy=False, add_humans=True):
        backend_config = {
            "scene_path": scene_path,
            "physics_config": "DEFAULT",
        }
        self.add_humans = add_humans
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

        self._done = True
        intrinsic_mat = self.get_intrinsics()
        intrinsic_mat_inv = np.linalg.inv(intrinsic_mat)
        img_resolution = self.get_img_resolution()
        img_pixs = np.mgrid[0 : img_resolution[0] : 1, 0 : img_resolution[1] : 1]
        img_pixs = img_pixs.reshape(2, -1)
        img_pixs[[0, 1], :] = img_pixs[[1, 0], :]
        uv_one = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))
        self.uv_one_in_cam = np.dot(intrinsic_mat_inv, uv_one)

    def restart_habitat(self):
        if hasattr(self, "_robot"):
            del self._robot
        backend_config = self.backend_config

        self._robot = Robot("habitat", common_config=backend_config, parent=self)
        from habitat_utils import reconfigure_scene

        # adds objects to the scene, doing scene-specific configurations
        reconfigure_scene(self, backend_config["scene_path"], self.add_humans)

    def get_habitat_state(self):
        """Returns the habitat position and rotation of the agent as lists"""
        sim = self._robot.base.sim
        agent = sim.get_agent(0)
        position = agent.get_state().position.tolist()
        rotation = quaternion.as_float_array(agent.get_state().rotation).tolist()
        return position, rotation

    def respawn_agent(self, position, rotation):
        """Respawns the agent at the position and rotation specified"""
        sim = self._robot.base.sim
        agent = sim.get_agent(0)
        new_agent_state = habitat_sim.AgentState()
        new_agent_state.position = position
        new_agent_state.rotation = quaternion.from_float_array(rotation)
        agent.set_state(new_agent_state)

    def test_connection(self):
        print("Connected!!")  # should print on server terminal
        return "Connected!"  # should print on client terminal

    def get_img_resolution(self):
        """return height and width"""
        return (512, 512)

    def get_pcd_data(self):
        """Gets all the data to calculate the point cloud for a given rgb, depth frame."""
        rgb, depth = self._robot.camera.get_rgb_depth()
        cur_state = self._robot.camera.agent.get_state()
        base_state = self.get_base_state()

        # cap anything more than np.power(2,6)~ 64 meter
        depth[depth > np.power(2, 6) - 1] = np.power(2, 6) - 1

        cur_sensor_state = cur_state.sensor_states["rgb"]
        initial_rotation = cur_state.rotation
        rot_init_rotation = self._robot.camera._rot_matrix(initial_rotation)
        relative_position = cur_sensor_state.position - cur_state.position
        relative_position = rot_init_rotation.T @ relative_position
        cur_rotation = self._robot.camera._rot_matrix(cur_sensor_state.rotation)
        cur_rotation = rot_init_rotation.T @ cur_rotation
        return rgb, depth, cur_rotation, -relative_position, base_state

    def get_current_pcd(self):
        rgb, depth, rot, trans, base_state = self.get_pcd_data()
        depth = depth.astype(np.float32)

        valid = depth > 0
        depth = depth[valid]
        rgb = rgb[valid]
        uv_one_in_cam = self.uv_one_in_cam[:, valid.reshape(-1)]

        depth = depth.reshape(-1)

        pts_in_cam = np.multiply(uv_one_in_cam, depth)
        pts_in_cam = np.concatenate((pts_in_cam, np.ones((1, pts_in_cam.shape[1]))), axis=0)
        pts = pts_in_cam[:3, :].T
        pts = np.dot(pts, rot.T)
        pts = pts + trans.reshape(-1)
        ros_to_habitat_frame = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        pts = ros_to_habitat_frame.T @ pts.T
        pts = pts.T
        pts = transform_pose(pts, base_state)

        return pts, rgb

    def get_open3d_pcd(self):
        pts, rgb = self.get_current_pcd()
        points, colors = pts.reshape(-1, 3), rgb.reshape(-1, 3)
        colors = colors / 255.0

        opcd = o3d.geometry.PointCloud()
        opcd.points = o3d.utility.Vector3dVector(points)
        opcd.colors = o3d.utility.Vector3dVector(colors)
        return opcd

    def is_obstacle_in_front(self, return_viz=False):
        base_state = self.get_base_state()
        pcd = self.get_open3d_pcd()
        ret = is_obstacle(pcd, base_state, max_dist=0.5, return_viz=return_viz)
        if return_viz:
            obstacle, cpcd, crop, bbox, rest = ret
            cpcd = o3d_pickle(cpcd)
            crop = o3d_pickle(crop)
            bbox = o3d_pickle(bbox)
            rest = o3d_pickle(rest)
            return obstacle, cpcd, crop, bbox, rest
        else:
            obstacle = ret
            return obstacle

    def go_to_absolute(
        self,
        xyt_position,
        wait=True,
    ):
        """Moves the robot base to given goal state in the world frame.

        :param xyt_position: The goal state of the form (x,y,yaw)
                             in the world (map) frame.

        :type xyt_position: list or np.ndarray
        """
        if self._done:
            self._done = False
            self._robot.base.go_to_absolute(xyt_position, wait=wait)
            self._done = True
        status = self.get_base_status()
        return status

    def go_to_relative(
        self,
        xyt_position,
        wait=True,
    ):
        """Moves the robot base to the given goal state relative to its current
        pose.

        :param xyt_position: The  relative goal state of the form (x,y,yaw)

        :type xyt_position: list or np.ndarray
        """
        if self._done:
            self._done = False
            self._robot.base.go_to_relative(xyt_position, wait=wait)
            self._done = True
        status = self.get_base_status()
        return status

    @Pyro4.oneway
    def stop(self):
        """stops robot base movement."""
        self._robot.base.stop()

    def get_base_state(self):
        """Returns the  base pose of the robot in the (x,y, yaw) format

        :return: pose of the form [x, y, yaw]
        :rtype: list
        """
        return self._robot.base.get_state()

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

    def get_rgb_depth_segm(self):
        return self._robot.camera.get_rgb_depth_segm()

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

    def is_busy(self):
        status = self._robot.base._as.get_state()
        return status == LocalActionStatus.ACTIVE

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
        "--scene_path",
        help="Optional config argument to be passed to the backend."
        "Currently mainly used to pass Habitat environment path",
        type=str,
        default="/Replica-Dataset/apartment_0/habitat/mesh_semantic.ply",
    )
    parser.add_argument(
        "--noisy",
        type=bool,
        default=os.getenv("NOISY_HABITAT", "False").lower() in ("true"),
        help="Set to True to load habitat with rgb, depth and movement noise models",
    )
    parser.add_argument(
        "--add_humans",
        type=bool,
        default=os.getenv("ADD_HUMANS", "True").lower() in ("true"),
        help="Set to True to load habitat without any humans",
    )

    args = parser.parse_args()

    np.random.seed(123)

    # GLContexts in general are thread local
    # The PyRobot <-> Habitat integration is not thread-aware / thread-configurable,
    # so our only option is to disable Pyro4's threading, and instead switch to
    # multiplexing (which isn't too bad)
    Pyro4.config.SERVERTYPE = "multiplex"

    with Pyro4.Daemon(args.ip) as daemon:
        robot = RemoteLocobot(
            scene_path=args.scene_path,
            noisy=args.noisy,
            add_humans=args.add_humans,
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
