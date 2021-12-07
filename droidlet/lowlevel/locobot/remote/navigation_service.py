import os
import random
import math
import time
import numpy as np
import Pyro4
from slam_pkg.utils import depth_util as du

random.seed(0)
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

class Trackback(object):
    def __init__(self, planner):
        self.locs = set()
        self.planner = planner

    def update(self, loc):
        self.locs.add(loc)

    def dist(self, a, b):
        a, b = a[:2], b[:2]
        d = np.linalg.norm((np.array(a) - np.array(b)), ord=1)
        return d

    def get_loc(self, cur_loc):
        ans = None
        d = 10000000
        # TODO: rewrite this logic to make it faster, by minimizing the number of
        # self.planner.get_short_term_goal calls
        # TODO: rewrite it as vectorized numpy
        cand = [x for x in self.locs if self.planner.get_short_term_goal(cur_loc, (x[1], x[0], 0))]
        for x in cand:
            if d > self.dist(cur_loc, x):
                ans = x
                d = self.dist(cur_loc, x)
        if ans is None:
            print("couldn't find a trackback location, tracking back to a random location")
            # if no trackback option found, then
            # try to trackback to a random location within a neighborhood
            # around current location
            x = cur_loc[0] + random.uniform(-0.2, 0.2)
            y = cur_loc[1] + random.uniform(-0.2, 0.2)
            angle = random.uniform(-math.pi, math.pi)
            ans = (x, y, angle)
        return ans


@Pyro4.expose
class Navigation(object):
    def __init__(self, planner, slam, robot):
        self.planner = planner
        self.slam = slam
        self.robot = robot
        self.trackback = Trackback(planner)
        self._busy = False

    def go_to_relative(self, goal):
        robot_loc = self.robot.get_base_state()
        abs_goal = du.get_relative_state(goal, (0.0, 0.0, -robot_loc[2]))
        abs_goal = list(abs_goal)
        abs_goal[0] += robot_loc[0]
        abs_goal[1] += robot_loc[1]
        abs_goal[2] = goal[2] + robot_loc[2]
        return self.go_to_absolute(abs_goal)
    
    def go_to_absolute(self, goal, steps=100000000):
        self._busy = True
        robot_loc = self.robot.get_base_state()
        initial_robot_loc = robot_loc
        goal_reached = False
        return_code = True
        while (not goal_reached) and steps > 0:
            stg = self.planner.get_short_term_goal(robot_loc, goal)
            if stg == False:
                # no path to end-goal
                return_code = False
                break
            self.robot.go_to_absolute(stg, wait=False)
            while self.robot.get_base_status() == "ACTIVE":
                time.sleep(0.01)
            robot_loc = self.robot.get_base_state()
            status = self.robot.get_base_status()
            print('go_to_absolute',
                  ' initial location: ', initial_robot_loc,
                  ' goal: ', goal,
                  ' short-term goal:', stg,
                  ' reached location: ', robot_loc,
                  ' robot status: ', status)
            if status == "SUCCEEDED":
                goal_reached = self.planner.goal_within_threshold(robot_loc, goal)
                self.trackback.update(robot_loc)
            else:
                # collided with something unexpected.
                robot_loc = self.robot.get_base_state()
                # Update SLAM map with an obstacle where collission occured
                # mark a point 5cm in front of the robot as obstacle
                collision_x = robot_loc[0] + 0.05 * np.cos(robot_loc[2])
                collision_y = robot_loc[1] + 0.05 * np.sin(robot_loc[2])
                collision_loc = (collision_x, collision_y, 0.)
                self.slam.add_obstacle(collision_loc)

                # trackback to a known good location
                trackback_loc = self.trackback.get_loc(robot_loc)

                print(f"Collided at {robot_loc}. Marking point {collision_loc} as obstacle."
                      f"Tracking back to {trackback_loc}")
                self.robot.go_to_absolute(trackback_loc, wait=True)
                # TODO: if the trackback fails, we're screwed. Handle this robustly.
            steps = steps - 1

        self._busy = False
        return return_code

    def explore(self, far_away_goal):
        if not hasattr(self, '_done_exploring'):
            self._done_exploring = False
        if not self._done_exploring:
            print("exploring 1 step")
            success = self.go_to_absolute(far_away_goal, steps=1)       
            if success == False:
                # couldn't reach far_away_goal
                # and don't seem to have any unexplored
                # paths to attempt to get there
                self._done_exploring = True
                print("exploration done")

    def is_busy(self):
        return self._busy

robot_ip = os.getenv('LOCOBOT_IP')
ip = os.getenv('LOCAL_IP')

with Pyro4.Daemon(ip) as daemon:
    robot = Pyro4.Proxy("PYRONAME:remotelocobot@" + robot_ip)
    planner = Pyro4.Proxy("PYRONAME:planner@" + robot_ip)
    slam = Pyro4.Proxy("PYRONAME:slam@" + robot_ip)

    obj = Navigation(planner, slam, robot)
    obj_uri = daemon.register(obj)
    with Pyro4.locateNS() as ns:
        ns.register("navigation", obj_uri)

    print("Navigation Server is started...")
    daemon.requestLoop()



