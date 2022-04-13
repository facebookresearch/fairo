import os
import sys
import time
import numpy as np
import Pyro4
import select
from slam_pkg.utils.map_builder import MapBuilder as mb
from slam_pkg.utils import depth_util as du
from skimage.morphology import disk, binary_dilation


Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


@Pyro4.expose
class SLAM(object):
    def __init__(
        self,
        robot,
        map_size=4000,
        resolution=5,
        robot_rad=30,
        agent_min_z=5,
        agent_max_z=70,
        obstacle_threshold=1,
    ):
        self.robot = robot
        self.robot_rad = robot_rad
        self.map_resolution = resolution
        self.map_builder = mb(
            map_size_cm=map_size,
            resolution=resolution,
            agent_min_z=agent_min_z,
            agent_max_z=agent_max_z,
            obs_thr=obstacle_threshold
        )
        self.map_size = map_size
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

        self.update_map()
        assert self.traversable is not None

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
        self.map_builder.add_obstacle(location)

    def update_map(self):
        pcd = self.robot.get_current_pcd()[0]
        self.map_builder.update_map(pcd)

        # explore the map by robot shape
        obstacle = self.map_builder.map[:, :, 1] >= 1.0
        selem = disk(self.robot_rad / self.map_builder.resolution)
        traversable = binary_dilation(obstacle, selem) != True
        self.traversable = traversable

    def get_map_resolution(self):
        return self.map_resolution

    def get_map(self):
        """returns the location of obstacles created by slam only for the obstacles,"""
        # get the index correspnding to obstacles
        indices = np.where(self.map_builder.map[:, :, 1] >= 1.0)
        # convert them into robot frame
        real_world_locations = [
            self.map2real([indice[0], indice[1]]).tolist()
            for indice in zip(indices[0], indices[1])
        ]
        return real_world_locations

    def reset_map(self, z_bins=None, obs_thr=None):
        self.map_builder.reset_map(self.map_size, z_bins=z_bins, obs_thr=obs_thr)


robot_ip = os.getenv("LOCOBOT_IP")
ip = os.getenv("LOCAL_IP")
robot_name = "remotelocobot"
if len(sys.argv) > 1:
    robot_name = sys.argv[1]
with Pyro4.Daemon(ip) as daemon:
    robot = Pyro4.Proxy("PYRONAME:" + robot_name + "@" + robot_ip)

    if robot_name == "hello_realsense":
        robot_height=141 # cm
        min_z = 20 # because of the huge spatial variance in realsense readings
        max_z = robot_height + 5 # cm
        obj = SLAM(
            robot,
            obstacle_threshold=10,
            agent_min_z=min_z,
            agent_max_z=max_z,
        )
    else:
        obj = SLAM(robot)
    obj_uri = daemon.register(obj)
    with Pyro4.locateNS(robot_ip) as ns:
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
