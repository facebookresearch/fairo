from re import I
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from slam_pkg.utils.depth_util import transform_pose, bin_points, splat_feat_nd
from slam_pkg.utils import depth_util as du


class MapBuilder(object):
    def __init__(
        self,
        map_size_cm=4000,
        resolution=5,
        obs_thr=1,
        cat_thr=5,
        agent_min_z=5,
        agent_max_z=70,
        num_semantic_categories=15,
    ):
        """
        :param map_size_cm: size of map in cm, assumes square map
        :param resolution: resolution of map, 1 pix = resolution distance(in cm) in real world
        :param obs_thr: number of depth points to be in bin to considered it as obstacle
        :param agent_min_z: robot min z (in cm), depth points below this will be considered as free space
        :param agent_max_z: robot max z (in cm), depth points above this will be considered as free space

        :type map_size_cm: int
        :type resolution: int
        :type obs_thr: int
        :type agent_min_z: int
        :type agent_max_z: int
        """
        self.map_size_cm = map_size_cm
        self.resolution = resolution
        self.obs_threshold = obs_thr
        self.cat_pred_threshold = cat_thr
        self.z_bins = [agent_min_z, agent_max_z]
        self.max_height = int(360 / self.resolution)
        self.min_height = int(-40 / self.resolution)
        self.map_size = int(self.map_size_cm // self.resolution)
        self.num_semantic_categories = num_semantic_categories

        self.map = np.zeros(
            (
                self.map_size,
                self.map_size,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )

        self.semantic_map = np.zeros(
            (
                self.num_semantic_categories + 1,
                self.map_size,
                self.map_size,
            ),
            dtype=np.float32,
        )

    def update_map(self, pcd, pose=None):
        """
        updated the map based on current observation (point cloud)
        :param pcd: point cloud in global frame, in meter

        :type pcd: np.ndarray [num_points, 3]
        :return: map of the environment, values [1-> obstacle, 0->free, unknown space]
        :rtype: np.ndarray
        """

        # convert point from m to cm
        pcd = pcd * 100

        # for mapping we want global center to be at origin
        geocentric_pc_for_map = transform_pose(
            pcd, (self.map_size_cm / 2.0, self.map_size_cm / 2.0, np.pi / 2.0)
        )
        geocentric_flat = bin_points(
            geocentric_pc_for_map, self.map.shape[0], self.z_bins, self.resolution
        )

        self.map = self.map + geocentric_flat

        map_gt = self.map[:, :, 1] / self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        return map_gt

    def update_semantic_map(self, pcd, semantic_channels):
        # convert point from m to cm
        pcd = pcd * 100

        # for mapping we want global center to be at origin
        geocentric_pc_for_map = transform_pose(
            pcd, (self.map_size_cm / 2.0, self.map_size_cm / 2.0, np.pi / 2.0)
        )

        geometric_pc_t = torch.from_numpy(geocentric_pc_for_map)

        geometric_pc_t[..., :2] = geometric_pc_t[..., :2] / self.resolution
        geometric_pc_t[..., :2] = (
            (geometric_pc_t[..., :2] - self.map_size // 2.0) / self.map_size * 2.0
        )
        max_h = self.max_height
        min_h = self.min_height
        geometric_pc_t[..., 2] = geometric_pc_t[..., 2] / self.resolution
        geometric_pc_t[..., 2] = (
            (geometric_pc_t[..., 2] - (max_h + min_h) // 2.0) / (max_h - min_h) * 2.0
        )
        geometric_pc_t = geometric_pc_t.transpose(0, 1).unsqueeze(0).float()

        init_grid = torch.zeros(
            1,
            semantic_channels.shape[1],
            self.map_size,
            self.map_size,
            self.max_height - self.min_height,
        ).float()

        feat = torch.from_numpy(semantic_channels.T).unsqueeze(0).float()
        feat[:, 0, :] = 1

        voxels_t = splat_feat_nd(init_grid, feat, geometric_pc_t).transpose(2, 3)

        top_down_map_t = voxels_t.sum(4)
        top_down_map = top_down_map_t.squeeze(0).numpy()

        self.semantic_map = self.semantic_map + top_down_map

        map_gt = np.copy(self.semantic_map)
        map_gt[0, :, :] = map_gt[0, :, :] / self.obs_threshold
        map_gt[1:, :, :] = map_gt[1:, :, :] / self.cat_pred_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        return map_gt

    def add_obstacle(self, location):
        self.map[round(location[1]), round(location[0]), 1] = 1

    def reset_map(self, map_size):
        """
        resets the map to unknown
        :param map_size: size of map in cm, assumes square map
        :type map_size: int
        """
        self.map_size_cm = map_size
        self.map_size = int(self.map_size_cm // self.resolution)

        self.map = np.zeros(
            (
                self.map_size,
                self.map_size,
                len(self.z_bins) + 1,
            ),
            dtype=np.float32,
        )

        self.semantic_map = np.zeros(
            (
                self.num_semantic_categories + 1,
                self.map_size,
                self.map_size,
            ),
            dtype=np.float32,
        )

    def get_map(self):
        """
        returns the map of the environment
        :return: 3 channel map of the environment, value [channel 1: points below agent_min_z,
        channel 2: points in between agent_min_z & agent_max_z, channel 3: points above agent_max_z ]
        :rtype: np.ndarray dim:[map_size, map_size, 3]
        """
        return self.map

    def real2map(self, loc):
        """
        convert real world location to map location
        :param loc: real world location in metric unit

        :type loc: tuple

        :return: location in map space
        :rtype: tuple [x_map_pix, y_map_pix]
        """
        # converts real location to map location
        loc = np.array([loc[0], loc[1], 0])
        loc *= 100  # convert location to cm
        map_loc = du.transform_pose(
            loc,
            (self.map_size_cm / 2.0, self.map_size_cm / 2.0, np.pi / 2.0),
        )
        map_loc /= self.resolution
        map_loc = map_loc.reshape(3)
        return tuple(map_loc[:2])

    def map2real(self, loc):
        """
        convert map location to real world location
        :param loc: map location [x_pixel_location, y_pixel_location]

        :type loc: list

        :return: corresponding map location in real world in metric unit
        :rtype: list [x_real_world, y_real_world]
        """
        # converts map location to real location
        loc = np.array([loc[0], loc[1], 0])
        real_loc = du.transform_pose(
            loc,
            (
                -self.map.shape[0] / 2.0,
                self.map.shape[1] / 2.0,
                -np.pi / 2.0,
            ),
        )
        real_loc *= self.resolution  # to take into account map resolution
        real_loc /= 100  # to convert from cm to meter
        real_loc = real_loc.reshape(3)
        return real_loc[:2]
