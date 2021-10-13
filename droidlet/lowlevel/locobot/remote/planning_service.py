import os
import numpy as np
import Pyro4
from slam_pkg.utils.fmm_planner import FMMPlanner


Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.ITER_STREAMING = True
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


# Pyro4.config.SERVERTYPE = "multiplex" # habitat

@Pyro4.expose
class Planning(object):
    def __init__(self, slam):
        self.slam = slam
        self.map_resolution = self.slam.get_map_resolution()

    def get_short_term_goal(self, robot_location, goal, step_size=25):
        """
        robot_location is simply get_base_state
        """

        # convert real co-ordinates to map co-ordinates
        goal_map_location = self.slam.real2map(goal[:2])

        # normalizes against initial robot state
        # if initial state wasn't (0,0,0)
        robot_map_location = self.slam.robot2map(robot_location)

        # get occupancy map
        traversable = self.slam.get_traversable_map()

        # construct a planner
        self.planner = FMMPlanner(
            traversable,
            step_size=int(step_size / self.map_resolution)
        )

        # set the goal and location in planner, get short-term-goal
        self.planner.set_goal(goal_map_location)
        stg = self.planner.get_short_term_goal(robot_map_location)

        # convert short-term-goal to real co-ordinates, and normalize
        # against robot initial state (if it wasn't zeros)
        stg_real = self.slam.map2robot(stg)
        return stg_real

    def goal_reached(self, robot_location, goal):
        goal_map_location = self.slam.real2map(goal[:2])
        robot_map_location = self.slam.robot2map(robot_location)

        distance = np.linalg.norm(np.array(robot_map_location)
                                  - np.array(goal_map_location)) * 100.0
        threshold = np.sqrt(2) * self.map_resolution

        return distance < threshold
    


robot_ip = os.getenv('LOCOBOT_IP')
ip = os.getenv('LOCAL_IP')
    
with Pyro4.Daemon(ip) as daemon:
    slam = Pyro4.Proxy("PYRONAME:slam@" + robot_ip)
    obj = Planning(slam)
    obj_uri = daemon.register(obj)
    with Pyro4.locateNS() as ns:
        ns.register("planning", obj_uri)

    print("Planning Server is started...")
    daemon.requestLoop()

