import os
import math
import numpy as np
import Pyro4
from slam_pkg.utils.fmm_planner import FMMPlanner
from rich import print

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
        self, robot_location, goal, step_size=25, distance_threshold=None, angle_threshold=None
    ):
        """
        robot_location is simply get_base_state
        """

        # convert real co-ordinates to map co-ordinates
        goal_map_location = self.slam.real2map(goal[:2])

        # normalizes against initial robot state
        # if initial state wasn't (0,0,0)
        robot_map_location = self.slam.robot2map(robot_location)

        # get occupancy map
        traversable_map = self.slam.get_traversable_map()

        # if the goal is an obstacle, you can't go there. Return
        if not is_traversable(goal_map_location, traversable_map):
            return False

        # construct a planner
        self.planner = FMMPlanner(traversable_map, step_size=int(step_size / self.map_resolution))

        # set the goal and location in planner, get short-term-goal
        self.planner.set_goal(goal_map_location)
        stg = self.planner.get_short_term_goal(robot_map_location)

        # if the goal is an obstacle, you can't go there. Return
        if not is_traversable(stg, traversable_map):
            return False

        # convert short-term-goal to real co-ordinates, and normalize
        # against robot initial state (if it wasn't zeros)
        stg_real = self.slam.map2robot(stg)

        if self.goal_within_threshold(stg_real, goal, distance_threshold, angle_threshold):
            # is it the final goal? if so,
            # the stg goes to within a 5cm resolution
            # -- related to the SLAM service's 2D map resolution.
            # so, finally, issue a last call to go to the precise final location
            # and to also use the rotation from the final goal
            print(
                "Short-term goal {} is within threshold or target goal {}".format(stg_real, goal)
            )
            print(f"This is the final goal, so returning the target goal directly {goal}")
            target_goal = goal
        else:
            print(
                "Short-term goal {} is not within threshold or target goal {}".format(
                    stg_real, goal
                )
            )
            rotation_angle = np.arctan2(
                stg_real[1] - robot_location[1], stg_real[0] - robot_location[0]
            )
            target_goal = (stg_real[0], stg_real[1], rotation_angle)
        return target_goal

    def goal_within_threshold(self, robot_location, goal, threshold=None, angle_threshold=None):
        if threshold is None:
            # in metres. map_resolution is the resolution of the SLAM's 2D map, so the planner can't
            # plan anything lower than this
            threshold = (float(self.map_resolution) - 1e-10) / 100.0
        if angle_threshold is None:
            angle_threshold = 1  # in degrees

        diff = np.abs(np.array(robot_location[:2]) - np.array(goal[:2]))
        distance = np.linalg.norm(diff)

        if len(robot_location) == 3 and len(goal) == 3:
            angle = robot_location[2] - goal[2]
            abs_angle = math.fabs(math.degrees(angle)) % 360

            within_threshold = (
                diff[0] < threshold and diff[1] < threshold and abs_angle < angle_threshold
            )
            print("goal_within_threshold: ", within_threshold)
            print(
                "Distance: x: {} < {}, y: {} < {}".format(diff[0], threshold, diff[1], threshold)
            )
            print("Angle: {} < {}".format(abs_angle, angle_threshold))
            print("Robot Location: {}".format(robot_location))
            print("Goal:           {}".format(goal))
        else:
            within_threshold = diff[0] < threshold and diff[1] < threshold
            print("goal_within_threshold: ", within_threshold)
            print(
                "Distance: x: {} < {}, y: {} < {}".format(diff[0], threshold, diff[1], threshold)
            )
            print("Robot Location: {}".format(robot_location))
            print("Goal:           {}".format(goal))
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
