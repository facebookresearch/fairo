import os
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
# Pyro4.config.ITER_STREAMING = True


# Pyro4.config.SERVERTYPE = "multiplex" # habitat


@Pyro4.expose
class SLAM(object):
    def __init__(self, robot,
                 map_size=4000,
                 resolution=5,
                 robot_rad=25,
                 agent_min_z=5,
                 agent_max_z=70,
    ):
        self.robot = robot
        self.robot_rad = robot_rad
        self.map_resolution = resolution
        self.map_builder = mb(
            map_size_cm=map_size,
            resolution=resolution,
            agent_min_z=agent_min_z,
            agent_max_z=agent_max_z,
        )
        self.init_state = self.robot.get_base_state()
        self.prev_bot_state = (0, 0, 0)

        self.update_map()
        assert self.traversable is not None

    def get_traversable_map(self):
        return self.traversable

    def real2map(self, real):
        return self.map_builder.real2map(real)
    
    def map2real(self, map_loc):
        return self.map_builder.map2real(map_loc)

    def robot2map(self, robot_loc):
        robot_location = du.get_relative_state(
            robot_loc,
            self.init_state)
        return self.real2map(robot_location)

    def update_map(self):
        robot_relative_pos = du.get_relative_state(
            self.robot.get_base_state(),
            self.init_state)
        pcd = self.robot.get_current_pcd(in_cam=False)[0]
        
        self.map_builder.update_map(pcd, robot_relative_pos)

        # explore the map by robot shape
        obstacle = self.map_builder.map[:, :, 1] >= 1.0
        selem = disk(self.robot_rad / self.map_builder.resolution)
        traversable = binary_dilation(obstacle, selem) != True
        self.traversable = traversable

    def get_map_resolution(self):
        return self.map_resolution


robot_ip = os.getenv('LOCOBOT_IP')
ip = os.getenv('LOCAL_IP')
with Pyro4.Daemon(ip) as daemon:
    robot = Pyro4.Proxy("PYRONAME:remotelocobot@" + robot_ip)
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

