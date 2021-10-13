import os
import numpy as np
import Pyro4
from slam_pkg.utils import depth_util as du

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.ITER_STREAMING = True
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


# Pyro4.config.SERVERTYPE = "multiplex" # habitat


@Pyro4.expose
class Navigation(object):
    def __init__(self, planner, robot):
        self.planner = planner
        self.robot = robot

    def go_to_relative(self, goal):
        robot_loc = self.robot.get_base_state()
        abs_goal = du.get_relative_state(goal, (0.0, 0.0, -robot_loc[2]))
        abs_goal = list(abs_goal)
        abs_goal[0] += robot_loc[0]
        abs_goal[1] += robot_loc[1]
        abs_goal[2] = goal[2] + robot_loc[2]

        print('start')
        print('goal', goal)
        print('abs_goal', abs_goal)
        self.go_to_absolute(abs_goal)

    
    def go_to_absolute(self, goal):
        print('goal:', goal)
        robot_loc = self.robot.get_base_state()
        goal_reached = False
        while not goal_reached:
            stg = self.planner.get_short_term_goal(robot_loc, goal)

            rotation_angle = np.arctan2(
                stg[1] - robot_loc[1],
                stg[0] - robot_loc[0]
            )
            print('rotation_angle', rotation_angle)

            self.robot.go_to_absolute((stg[0], stg[1], rotation_angle))
            
            robot_loc = self.robot.get_base_state()
            goal_reached = self.planner.goal_reached(robot_loc, goal)

    # slam wrapper
    def explore(self):
        pass

    def pause_explore(self):
        pass

    def is_busy(self):
        pass


robot_ip = os.getenv('LOCOBOT_IP')
ip = os.getenv('LOCAL_IP')

with Pyro4.Daemon(ip) as daemon:
    robot = Pyro4.Proxy("PYRONAME:remotelocobot@" + robot_ip)
    planner = Pyro4.Proxy("PYRONAME:planning@" + robot_ip)

    obj = Navigation(planner, robot)
    obj_uri = daemon.register(obj)
    with Pyro4.locateNS() as ns:
        ns.register("navigation", obj_uri)

    print("Navigation Server is started...")
    daemon.requestLoop()



