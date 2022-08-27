# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import pyrobot.utils.util as prutil

import habitat_sim.utils as habUtils
from .transformations import euler_from_quaternion, euler_from_matrix


class LoCoBotCamera(object):
    """docstring for SimpleCamera"""

    def __init__(self, configs, simulator):
        self.sim = simulator.sim
        self.configs = configs
        self.agent = self.sim.get_agent(self.configs.COMMON.SIMULATOR.DEFAULT_AGENT_ID)

        # Pan and tilt related vairades.
        self.pan = 0.0
        self.tilt = 0.0

    def get_rgb(self):
        observations = self.sim.get_sensor_observations()
        return observations["rgb"][:, :, 0:3]

    def get_depth(self):
        observations = self.sim.get_sensor_observations()
        return observations["depth"] / self.configs.CAMERA.DEPTH_MAP_FACTOR

    def get_rgb_depth(self):
        observations = self.sim.get_sensor_observations()
        return (
            observations["rgb"][:, :, 0:3],
            observations["depth"] / self.configs.CAMERA.DEPTH_MAP_FACTOR,
        )

    def get_rgb_depth_segm(self):
        observations = self.sim.get_sensor_observations()
        return (
            observations["rgb"][:, :, 0:3],
            observations["depth"] / self.configs.CAMERA.DEPTH_MAP_FACTOR,
            observations["semantic"],
        )

    def get_intrinsics(self):
        """
        Returns the instrinsic matrix of the camera

        :return: the intrinsic matrix (shape: :math:`[3, 3]`)
        :rtype: np.ndarray
        """
        height, width = self.configs.COMMON.SIMULATOR.AGENT.SENSORS.RESOLUTIONS[0]
        hfov = math.radians(self.configs.COMMON.SIMULATOR.AGENT.SENSORS.HFOVS[0])
        vfov = 2 * math.atan(math.tan(hfov / 2) * height / width)

        # https://github.com/facebookresearch/habitat-lab/issues/656
        # https://github.com/facebookresearch/habitat-lab/issues/474
        # https://github.com/facebookresearch/habitat-lab/issues/499
        fx = width / math.tan(hfov / 2) / 2
        fy = height / math.tan(vfov / 2) / 2
        cx = width / 2
        cy = height / 2
        Itc = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return Itc

    def _rot_matrix(self, habitat_quat):
        quat_list = [habitat_quat.x, habitat_quat.y, habitat_quat.z, habitat_quat.w]
        return prutil.quat_to_rot_mat(quat_list)

    @property
    def state(self):
        """
        Return the current pan and tilt joint angles of the robot camera.

        :return:
                pan_tilt: A list the form [pan angle, tilt angle]
        :rtype: list
        """
        return self.get_state()

    def get_state(self):
        """
        Return the current pan and tilt joint angles of the robot camera.

        :return:
                pan_tilt: A list the form [pan angle, tilt angle]
        :rtype: list
        """
        return [self.pan, self.tilt]

    def get_pan(self):
        """
        Return the current pan joint angle of the robot camera.

        :return:
                pan: Pan joint angle
        :rtype: float
        """
        return self.pan

    def get_tilt(self):
        """
        Return the current tilt joint angle of the robot camera.

        :return:
                tilt: Tilt joint angle
        :rtype: float
        """
        return self.tilt

    def set_pan(self, pan, wait=True):
        """
        Sets the pan joint angle to the specified value.

        :param pan: value to be set for pan joint
        :param wait: wait until the pan angle is set to
                     the target angle.

        :type pan: float
        :type wait: bool
        """

        self.set_pan_tilt(pan, self.tilt)

    def set_tilt(self, tilt, wait=True):
        """
        Sets the tilt joint angle to the specified value.

        :param tilt: value to be set for the tilt joint
        :param wait: wait until the tilt angle is set to
                     the target angle.

        :type tilt: float
        :type wait: bool
        """

        self.set_pan_tilt(self.pan, tilt)

    def _compute_relative_pose(self, pan, tilt):
        pan_link = 0.1  # length of pan link
        tilt_link = 0.1  # length of tilt link

        sensor_offset_tilt = np.asarray([0.0, 0.0, -1 * tilt_link])

        quat_cam_to_pan = habUtils.quat_from_angle_axis(-1 * tilt, np.asarray([1.0, 0.0, 0.0]))

        sensor_offset_pan = habUtils.quat_rotate_vector(quat_cam_to_pan, sensor_offset_tilt)
        sensor_offset_pan += np.asarray([0.0, pan_link, 0.0])

        quat_pan_to_base = habUtils.quat_from_angle_axis(-1 * pan, np.asarray([0.0, 1.0, 0.0]))

        sensor_offset_base = habUtils.quat_rotate_vector(quat_pan_to_base, sensor_offset_pan)
        # sensor_offset_base += np.asarray([0.0, 0.5, 0.1])  # offset w.r.t base
        # sensor_offset_base += np.asarray([0.0, 1.31, 0.1])  # offset w.r.t base
        sensor_offset_base += np.asarray([0.0, 0.88, 0.1])  # offset w.r.t base

        # translation
        quat = quat_cam_to_pan * quat_pan_to_base
        return sensor_offset_base, quat.inverse()

    def set_pan_tilt(self, pan, tilt, wait=True):
        """
        Sets both the pan and tilt joint angles to the specified values.

        :param pan: value to be set for pan joint
        :param tilt: value to be set for the tilt joint
        :param wait: wait until the pan and tilt angles are set to
                     the target angles.

        :type pan: float
        :type tilt: float
        :type wait: bool
        """
        self.pan = pan
        self.tilt = tilt
        sensor_offset, quat_base_to_sensor = self._compute_relative_pose(pan, tilt)
        cur_state = self.agent.get_state()  # Habitat frame
        sensor_position = cur_state.position + sensor_offset
        sensor_quat = cur_state.rotation * quat_base_to_sensor

        for sensor in cur_state.sensor_states:
            cur_state.sensor_states[sensor].position = sensor_position
            cur_state.sensor_states[sensor].rotation = sensor_quat

        self.agent.set_state(cur_state, reset_sensors=False, infer_sensor_states=False)

    def reset(self):
        """
        This function resets the pan and tilt joints by actuating
        them to their home configuration.
        """
        self.set_pan_tilt(self.configs.CAMERA.RESET_PAN, self.configs.CAMERA.RESET_TILT)
