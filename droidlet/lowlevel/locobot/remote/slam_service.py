import os
import sys
import time
import random
import torch
import torch.nn as nn
import numpy as np
import Pyro4
import select
from rich import print

from slam_pkg.utils.map_builder import MapBuilder as mb
from slam_pkg.utils import depth_util as du
from skimage.morphology import disk, binary_dilation
from segmentation.constants import coco_categories
from rich import print
from droidlet.lowlevel.pyro_utils import safe_call

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


@Pyro4.expose
class SLAM(object):
    def __init__(
        self,
        robot,
        map_size_cm=2400,
        resolution=5,
        robot_rad=20,
        agent_min_z=5,
        agent_max_z=70,
        obstacle_threshold=1,
        category_threshold=5,
        global_downscaling=2,
    ):
        self.robot = robot
        self.robot_rad = robot_rad
        self.map_resolution = resolution
        self.num_sem_categories = len(coco_categories)
        self.obs_threshold = obstacle_threshold
        self.map_builder = mb(
            pose_init=self.robot.get_base_state(),
            map_size_cm=map_size_cm,
            resolution=resolution,
            agent_min_z=agent_min_z,
            agent_max_z=agent_max_z,
            obs_thr=obstacle_threshold,
            cat_thr=category_threshold,
            num_sem_categories=self.num_sem_categories,
        )
        self.map_size_cm = map_size_cm
        self.map_size = int(self.map_size_cm // self.map_resolution)
        self.global_downscaling = global_downscaling
        self.local_map_size = self.map_size // global_downscaling
        # if the map is a previous map loaded from disk, and
        # if the robot looks around and registers itself at a
        # non-origin location in the map just as it is coming up,
        # then the robot's reported origin (from get_base_state) is
        # not the map's origin. In such cases, `self.init_state`
        # is useful, as it is used to handle all co-ordinate transforms
        # correctly.
        # Currently, self.init_state is kinda useless and not utilized
        # in any meaningful way
        self.init_state = (0.0, 0.0, 0.0)
        self.prev_bot_state = (0.0, 0.0, 0.0)

        self.last_position_vis_info = None
        self.update_semantic_map = True

        self.update_map()
        assert self.traversable is not None

    def disable_semantic_map_update(self):
        self.update_semantic_map = False

    def get_map_sizes(self):
        return self.map_size, self.local_map_size

    def get_traversable_map(self):
        return self.traversable

    def real2map(self, real):
        return self.map_builder.real2map(real)

    def map2real(self, map_loc):
        return self.map_builder.map2real(map_loc)

    def robot2map(self, robot_loc):
        # TODO: re-enable this code when init_state can be non-zero
        # robot_location = du.get_relative_state(
        #     robot_loc,
        #     self.init_state)
        return self.real2map(robot_loc)

    def map2robot(self, map_loc):
        return self.map2real(map_loc)
        # TODO: re-enable and test this code when init_state can be non-zero
        # real_loc = self.map2real(map_loc)
        # loc = du.get_relative_state(real_loc, (0.0, 0.0, -self.init_state[2]))

        # # 2) add the offset
        # loc = list(loc)
        # loc[0] += self.init_state[0]
        # loc[1] += self.init_state[1]
        # return tuple(loc)

    def add_obstacle(self, location, in_map=False):
        """
        add an obstacle at the given location.
        if in_map=False, then location is given in real co-ordinates
        if in_map=True, then location is given in map co-ordinates
        """
        if not in_map:
            location = self.real2map(location)
        print(f"Adding obstacle at {location}!")
        self.map_builder.add_obstacle(location)

    def add_obstacles(self, locations, in_map=False):
        """
        add a list of obstacles at the given locations.
        if in_map=False, then location is given in real co-ordinates
        if in_map=True, then location is given in map co-ordinates
        """
        for loc in locations:
            self.add_obstacle(loc, in_map=in_map)

    def update_map(self):
        pcd, rgb, depth = self.robot.get_current_pcd()
        pose = self.robot.get_base_state()

        self.map_builder.update_map(pcd)

        if self.update_semantic_map:
            t0 = time.time()
            semantics, unfiltered_semantics, semantics_vis = self.robot.get_semantics(rgb.copy())#, depth.copy())
            t1 = time.time()
            print("get_semantics()", t1 - t0)

            self.map_builder.update_semantic_map(pcd, semantics, pose)
            t2 = time.time()
            print("update_semantic_map()", t2 - t1)

            # TODO Temporary way to pass this point cloud to the pick and place service
            manipulation_pcd = self.robot.get_manipulation_pcd_from_depth(depth)

            self.last_position_vis_info = {
                # Frame
                "rgb": rgb,
                "depth": depth,
                "pcd": pcd,
                "manipulation_pcd": manipulation_pcd,
                "semantic_frame": semantics,
                "unfiltered_semantic_frame": unfiltered_semantics,
                "semantic_frame_vis": semantics_vis,
                # Semantic map after update with frame
                "semantic_map": self.get_global_semantic_map(),
                # Sensor pose of frame
                "pose": pose,
            }

        # explore the map by robot shape
        obstacle = self.map_builder.map[:, :, 1] >= self.obs_threshold
        selem = disk(self.robot_rad / self.map_builder.resolution)
        traversable = binary_dilation(obstacle, selem) != True
        self.traversable = traversable

    def get_map_resolution(self):
        return self.map_resolution

    def get_map(self):
        """returns the location of obstacles created by slam only for the obstacles,"""
        # get the index correspnding to obstacles
        indices = np.where(self.map_builder.map[:, :, 1] >= self.obs_threshold)
        # convert them into robot frame
        real_world_locations = [
            self.map2real([indice[0], indice[1]]).tolist()
            for indice in zip(indices[0], indices[1])
        ]
        return real_world_locations

    def get_last_position_vis_info(self):
        return self.last_position_vis_info

    def get_global_semantic_map(self):
        return self.map_builder.semantic_map

    def get_local_semantic_map(self):
        global_map = self.get_global_semantic_map()
        x, y, _ = self.robot.get_base_state()
        c, r = self.map_builder.real2map((x, y))

        c1 = max(int(c - self.local_map_size // 2), 0)
        r1 = max(int(r - self.local_map_size // 2), 0)
        if c1 + self.local_map_size > self.map_size:
            c2 = self.map_size
            c1 = c2 - self.local_map_size
        else:
            c2 = c1 + self.local_map_size
        if r1 + self.local_map_size > self.map_size:
            r2 = self.map_size
            r1 = r2 - self.local_map_size
        else:
            r2 = r1 + self.local_map_size

        local_map = global_map[:, r1:r2, c1:c2]
        return local_map

    def get_semantic_map_features(self):
        """
        Returns:
            map_features: semantic map features
            orientation: discretized yaw in {0, ..., 72}
        """
        global_map = torch.from_numpy(self.get_global_semantic_map())
        local_map = torch.from_numpy(self.get_local_semantic_map())

        map_features = torch.zeros(
            self.num_sem_categories + 8, self.local_map_size, self.local_map_size
        )
        # Local obstacles, explored area, and current and past position
        map_features[:4, :, :] = local_map[:4, :, :]
        # Global obstacles, explored area, and current and past position
        map_features[4:8, :, :] = nn.MaxPool2d(self.global_downscaling)(global_map[:4, :, :])
        # Local semantic categories
        map_features[8:, :, :] = local_map[4:, :, :]

        return map_features.unsqueeze(0)

    def get_orientation(self):
        """Get discretized robot orientation."""
        return self.robot.get_orientation()

    def reset_map(self, z_bins=None, obs_thr=None):
        self.map_builder.reset_map(self.map_size_cm, z_bins=z_bins, obs_thr=obs_thr)


robot_ip = os.getenv("LOCOBOT_IP")
ip = os.getenv("LOCAL_IP")
robot_name = "remotelocobot"
if len(sys.argv) > 1:
    robot_name = sys.argv[1]
with Pyro4.Daemon(ip) as daemon:
    robot = Pyro4.Proxy("PYRONAME:" + robot_name + "@" + robot_ip)

    if robot_name == "hello_realsense":
        robot_height = 141  # cm
        min_z = 20  # because of the huge spatial variance in realsense readings
        max_z = robot_height + 5  # cm
        obj = SLAM(
            robot,
            obstacle_threshold=300,  # 300 on robot, 5 in sim
            agent_min_z=min_z,
            agent_max_z=max_z,
        )
    else:
        obj = SLAM(robot)
    obj_uri = daemon.register(obj)
    with Pyro4.locateNS(host=robot_ip) as ns:
        ns.register("slam", obj_uri)

    print("SLAM Server is started...")

    def refresh():
        obj.update_map()
        # print("In refresh: ", time.asctime())
        return True

    daemon.requestLoop(refresh)

    # visit this later
    # try:
    #     while True:
    #         print(time.asctime(), "Waiting for requests...")

    #         sockets = daemon.sockets
    #         ready_socks = select.select(sockets, [], [], 0)
    #         events = []
    #         for s in ready_socks:
    #             events.append(s)
    #         daemon.events(events)
    # except KeyboardInterrupt:
    #     pass
