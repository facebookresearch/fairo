import os
import math
import numpy as np
import random
import Pyro4
from slam_pkg.utils.fmm_planner import FMMPlanner
from rich import print

random.seed(0)
np.random.seed(0)
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


def is_traversable(location, traversable):
    return traversable[round(location[1]), round(location[0])]


@Pyro4.expose
class Planner(object):
    def __init__(self, slam):
        self.slam = slam
        self.map_resolution = self.slam.get_map_resolution()

    def get_short_term_goal(
        self,
        robot_location,
        goal=None,
        goal_map=None,
        step_size=25,
        vis_path=None,
    ):
        """
        Args:
            robot_location: simply get_base_state()
            goal: (x, y) real world goal
            goal_map: binary map that contains multiple goals (encoded as 1)
        """
        # specify exactly one of goal or goal_map
        assert (goal is not None and goal_map is None) or (goal is None and goal_map is not None)

        # normalizes against initial robot state
        # if initial state wasn't (0,0,0)
        robot_map_location = self.slam.robot2map(robot_location)

        # get occupancy map
        traversable_map = self.slam.get_traversable_map()

        # construct a planner
        self.planner = FMMPlanner(traversable_map, step_size=int(step_size / self.map_resolution))

        if goal_map is not None:
            # TODO Is it necessary to check that at least one goal in the goal map is reachable?
            self.planner.set_multi_goal(goal_map, vis_path=vis_path)

        else:
            # convert robot co-ordinates to map co-ordinates
            goal_map_location = self.slam.robot2map(goal[:2])

            # if the goal is an obstacle, you can't go there; return
            if not is_traversable(goal_map_location, traversable_map):
                return False

            # set the goal location in planner
            self.planner.set_goal(goal_map_location, vis_path=vis_path)

        # get short term goal
        stg = self.planner.get_short_term_goal(robot_map_location)

        # convert short-term-goal to real co-ordinates, and normalize
        # against robot initial state (if it wasn't zeros)
        stg_real = self.slam.map2robot(stg)

        if goal_map is not None:
            rotation_angle = np.arctan2(
                stg_real[1] - robot_location[1], stg_real[0] - robot_location[0]
            )
            target_goal = (stg_real[0], stg_real[1], rotation_angle)
        else:
            # Hack to get panorama start working - this is probably not the right thing
            # to do in general
            target_goal = (stg_real[0], stg_real[1], goal[2])

        return target_goal

    def goal_within_threshold(
        self,
        robot_location,
        goal=None,
        goal_map=None,
        distance_threshold=None,
        angle_threshold=None,
    ):
        # specify exactly one of goal or goal_map
        assert (goal is not None and goal_map is None) or (goal is None and goal_map is not None)

        if angle_threshold is None:
            # in degrees
            # angle_threshold = 1
            angle_threshold = 30

        if distance_threshold is None:
            # in metres
            # map_resolution is the resolution of the SLAM's 2D map, so the planner can't
            # plan anything lower than this
            # distance_threshold = (float(self.map_resolution) - 1e-10) / 100.0
            distance_threshold = 0.5

        if goal_map is not None:
            # check whether the robot is within threshold of the closest goal in the goal map
            robot_map_location = self.slam.robot2map(robot_location)
            goal_locations = np.transpose(np.nonzero((goal_map.T == 1)))
            distances = np.linalg.norm(goal_locations - robot_map_location, axis=1)
            goal_in_map = goal_locations[distances.argmin()]
            goal = self.slam.map2robot(goal_in_map)

        diff = np.abs(np.array(robot_location[:2]) - np.array(goal[:2]))
        distance = np.linalg.norm(diff)

        if len(robot_location) == 3 and len(goal) == 3:
            angle = robot_location[2] - goal[2]
            abs_angle = math.fabs(math.degrees(angle)) % 360
            abs_angle = min(abs_angle, 360 - abs_angle)

            within_threshold = (
                diff[0] < distance_threshold
                and diff[1] < distance_threshold
                and abs_angle < angle_threshold
            )
            print("goal_within_threshold: ", within_threshold)
            print(f"Distance: {distance} < {distance_threshold}")
            print(f"Angle: {abs_angle} < {angle_threshold}")
            print(f"Robot Location: {robot_location}")
            print(f"Goal:           {goal}")

        else:
            within_threshold = diff[0] < distance_threshold and diff[1] < distance_threshold
            print("goal_within_threshold: ", within_threshold)
            print(f"Distance: {distance} < {distance_threshold}")
            print(f"Robot Location: {robot_location}")
            print(f"Goal:           {goal}")

        return within_threshold


robot_ip = os.getenv("LOCOBOT_IP")
ip = os.getenv("LOCAL_IP")

with Pyro4.Daemon(ip) as daemon:
    slam = Pyro4.Proxy("PYRONAME:slam@" + robot_ip)
    obj = Planner(slam)
    obj_uri = daemon.register(obj)
    with Pyro4.locateNS(host=robot_ip) as ns:
        ns.register("planner", obj_uri)

    print("Planner Server is started...")
    daemon.requestLoop()
