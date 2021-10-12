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
        self.init_state = self.robot.get_base_state()

    def get_rel_state(self, cur_state, init_state):
        # get relative in global frame
        rel_X = cur_state[0] - init_state[0]
        rel_Y = cur_state[1] - init_state[1]
        # transfer from global frame to init frame
        R = np.array(
            [
                [np.cos(init_state[2]), np.sin(init_state[2])],
                [-np.sin(init_state[2]), np.cos(init_state[2])],
            ]
        )
        rel_x, rel_y = np.matmul(R, np.array([rel_X, rel_Y]).reshape(-1, 1))
        return rel_x[0], rel_y[0], cur_state[2] - init_state[2]

    def get_absolute_goal(self, loc):
        """
        Transfer location in init robot frame to global frame
        :param loc: location in init frame in metric unit

        :type loc: tuple

        :return: location in global frame in metric unit
        :rtype: list
        """
        # 1) orient goal to global frame
        loc = du.get_relative_state(loc, (0.0, 0.0, -self.init_state[2]))

        # 2) add the offset
        loc = list(loc)
        loc[0] += self.init_state[0]
        loc[1] += self.init_state[1]
        return tuple(loc)


    def go_to_relative(self, goal):
        robot_pr_pose = self.robot.get_base_state()
        abs_pr_goal = self.get_rel_state(goal, (0.0, 0.0, -robot_pr_pose[2]))
        abs_pr_goal = list(abs_pr_goal)
        abs_pr_goal[0] += robot_pr_pose[0]
        abs_pr_goal[1] += robot_pr_pose[1]
        abs_pr_goal[2] = goal[2] + robot_pr_pose[2]

        print('start')
        print('init_state', self.init_state)
        print('goal', goal)
        print('abs_pr_goal', abs_pr_goal)
        # convert the goal in init frame
        goal_loc = self.get_rel_state(abs_pr_goal, self.init_state)

        print('abs_goal', goal_loc)
        
        self.go_to_absolute(goal_loc)

    
    def go_to_absolute(self, goal):
        goal = du.get_relative_state(goal, self.init_state)
        print('goal:', goal)
        robot_loc = self.robot.get_base_state()
        goal_reached = False
        while not goal_reached:
            stg = self.planner.get_short_term_goal(robot_loc, goal)
            stg = self.get_absolute_goal((stg[0], stg[1], 0))

            rotation_angle = np.arctan2(
                stg[1] - robot_loc[1],
                stg[0] - robot_loc[0]
            ) + self.init_state[2]
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



