# need to convert it to api
from pyrobot import Robot
from pyrobot.utils.util import try_cv2_import
from pyrobot.locobot.base_control_utils import LocalActionStatus

import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import argparse
from scipy import ndimage
from copy import deepcopy as copy
import time
from math import ceil, floor, radians
import sys
import json
import random
import shutil

cv2 = try_cv2_import()

# for slam modules
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from skimage.morphology import disk, binary_dilation
from slam_pkg.utils.map_builder import MapBuilder as mb
from slam_pkg.utils.fmm_planner import FMMPlanner
from slam_pkg.utils import depth_util as du

class TrackBack(object):
    def __init__(self):
        self.locs = set()

    def update(self, loc):
        self.locs.add(loc)

    def dist(self, a, b):
        d = np.linalg.norm((np.array(a) - np.array(b)), ord=1)
        # print(f'dist {a, b} = {d}')
        return d

    def get_loc(self, cur_loc, traversable):
        ans = None
        d = 10000000
        for x in self.locs:
            if not traversable[round(x[1]), round(x[0])]:
                print(f'removing {x} not traversable')
                self.locs.remove(x)
                continue
            # print(f'candidate loc {round(x[0]), round(x[1])}, cur_loc {cur_loc}')
            if d > self.dist(cur_loc, x):
                ans = x
                d = self.dist(cur_loc, x)
        print(f'track back loc {ans}')
        return ans

class Slam(object):
    def __init__(
        self,
        robot,
        map_size=4000,
        resolution=5,
        robot_rad=25,
        agent_min_z=5,
        agent_max_z=70,
    ):
        """

        :param robot: pyrobot robot object, only supports [habitat, locobot]
        :param map_size: size of map to be build in cm, assumes square map
        :param resolution: resolution of map, 1 pix = resolution distance(in cm) in real world
        :param robot_rad: radius of the agent, used to explode the map
        :param agent_min_z: robot min z (in cm), depth points below this will be considered as free space
        :param agent_max_z: robot max z (in cm), depth points above this will be considered as free space

        :type robot: pytobot.Robot
        :type map_size: int
        :type resolution: int
        :type robot_rad: int
        :type agent_min_z: int
        :type agent_max_z: int
        """
        self.robot = robot
        self.robot_rad = robot_rad
        self.map_builder = mb(
            map_size_cm=map_size,
            resolution=resolution,
            agent_min_z=agent_min_z,
            agent_max_z=agent_max_z,
        )

        # initialize variable
        robot.camera.reset()
        time.sleep(2)

        self.init_state = self.get_robot_global_state()
        self.prev_bot_state = (0, 0, 0)
        self.col_map = np.zeros((self.map_builder.map.shape[0], self.map_builder.map.shape[1]))
        self.robot_loc_list_map = np.array(
            [
                self.real2map(
                    du.get_relative_state(self.get_robot_global_state(), self.init_state)[:2]
                )
            ]
        )
        self.map_builder.update_map(
            self.robot.camera.get_current_pcd(in_cam=False)[0],
            du.get_relative_state(self.get_robot_global_state(), self.init_state),
        )

        
        self.last_pos = self.robot.base.get_state()

        self.whole_area_explored = True
        self.last_stg = None
        self.explore_goal = None
        self.debug_state = {}
        self.track_back = TrackBack()


    def set_explore_goal(self, goal):
        print(f'setting explore goal {goal}')
        self.explore_goal = goal

    def set_goal(self, goal):
        """
        goal is 3 len tuple with position in real world in robot start frame
        :param goal: goal to be reached in metric unit

        :type goal: tuple

        :return:
        """
        self.goal_loc = goal
        self.goal_loc_map = self.real2map(self.goal_loc[:2])
        print(f'set_goal {self.goal_loc, self.goal_loc_map, goal}')

    def set_relative_goal_in_robot_frame(self, goal):
        """
        goal is 3 len tuple with position in real world in robot current frmae
        :param goal: goal to be reached in metric unit

        :type goal: tuple

        :return:
        """
        robot_pr_pose = self.get_robot_global_state()
        # check this part
        abs_pr_goal = list(du.get_relative_state(goal, (0.0, 0.0, -robot_pr_pose[2])))
        abs_pr_goal[0] += robot_pr_pose[0]
        abs_pr_goal[1] += robot_pr_pose[1]
        abs_pr_goal[2] = goal[2] + robot_pr_pose[2]

        # convert the goal in init frame
        self.goal_loc = du.get_relative_state(abs_pr_goal, self.init_state)
        self.goal_loc_map = self.real2map(self.goal_loc[:2])

        # TODO: make it non blocking
        while self.take_step(25) is None:
            print(f'set_relative_goal_in_robot_frame')
            continue

    def set_absolute_goal_in_robot_frame(self, goal):
        """
        goal is 3 len tuple with position in real world in robot start frmae
        :param goal: goal to be reached in metric unit

        :type goal: tuple

        :return:
        """
        # convert the relative goal to abs goal
        self.goal_loc = du.get_relative_state(goal, self.init_state)
        # convert the goal in inti frame
        self.goal_loc_map = self.real2map(self.goal_loc[:2])
        print(f'set_absolute_goal_in_robot_frame {self.goal_loc, self.goal_loc_map, goal}')
        # TODO make it non blocking
        while self.take_step(25) is None:
            print(f'set_absolute_goal_in_robot_frame')
            continue

    def update_map(self):
        """Updtes map , explode it by the radius of robot, add collison map to it and return the traversible area

        Returns:
            [np.ndarray]: [traversible space]
        """
        robot_state = du.get_relative_state(self.get_robot_global_state(), self.init_state)
        self.map_builder.update_map(
            self.robot.camera.get_current_pcd(in_cam=False)[0], robot_state
        )

        # explore the map by robot shape
        obstacle = self.map_builder.map[:, :, 1] >= 1.0
        selem = disk(self.robot_rad / self.map_builder.resolution)
        traversable = binary_dilation(obstacle, selem) != True

        return traversable
    
    def get_stg(self, step_size):
        traversable = self.update_map()
        self.planner = FMMPlanner(
            traversable, step_size=int(step_size / self.map_builder.resolution)
        )
        self.planner.set_goal(self.goal_loc_map)
        robot_map_loc = self.real2map(
            du.get_relative_state(self.get_robot_global_state(), self.init_state)
        )
        self.stg = self.planner.get_short_term_goal(robot_map_loc)
        return traversable

    def take_step(self, step_size):
        """
        step size in meter
        :param step_size:
        :return:
        """
        print(f'\nstep begin ...')
        traversable = self.get_stg(step_size) # sets self.stg

        # print(f'self.goal_loc {self.goal_loc} self.explore_goal {self.explore_goal}')
        # convert goal from map space to robot space
        stg_real = self.map2real(self.stg)

        def pp(n, t):
            print(f'{n}, {round(t[0]), round(t[1])} ')

        pp('goal_loc', self.goal_loc)
        pp('goal_loc_map', self.goal_loc_map)
        pp('stg', self.stg)

        # convert stg real from init frame to global frame of pyrobot
        stg_real_g = self.get_absolute_goal((stg_real[0], stg_real[1], 0))
        robot_state = du.get_relative_state(self.get_robot_global_state(), self.init_state)
        robot_map_loc = self.real2map(robot_state)

        pp(f'robot_map_loc before translation', robot_map_loc)
        print(f'self.planner.fmm_dist[{self.stg[1]}][{self.stg[0]}] = {self.planner.fmm_dist[self.stg[1], self.stg[0]]}')
        # print(f'self.planner.fmm_dist[{self.stg[0]}][{self.stg[1]}] = {self.planner.fmm_dist[self.stg[0], self.stg[1]]}')

        # check whether goal is on collision # should never happen? 
        if not traversable[self.stg[1], self.stg[0]]:
            print("Obstacle in path! Should never happen, stg should never be an obstacle!!")
            print(f'traversable[stg] {traversable[self.stg[1], self.stg[0]]}')
            # print(f'robot_map_loc {robot_map_loc} traversable.shape {traversable.shape}')
            print(f'traversable[robot_loc] {traversable[round(robot_map_loc[1]), round(robot_map_loc[0])]}')

        else:            
            # go to the location the robot
            exec = self.robot.base.go_to_absolute(
                (
                    stg_real_g[0],
                    stg_real_g[1],
                    np.arctan2(
                        stg_real[1] - self.prev_bot_state[1], stg_real[0] - self.prev_bot_state[0] #stg_real[1] - rs[1], stg_real[0] - rs[0]
                    ) + self.init_state[2],
                ),
                wait=self.exec_wait,
            )
            while self.robot.base._as.get_state() == LocalActionStatus.ACTIVE:
                pass
        
        exec = self.robot.base._as.get_state() == LocalActionStatus.SUCCEEDED
        robot_map_loc = self.real2map(
            du.get_relative_state(self.get_robot_global_state(), self.init_state)
        )
        if exec:
            # if self.robot.base._as.get_state() == LocalActionStatus.SUCCEEDED:
            print(f'finished translation')
            pp('robot_map_loc', robot_map_loc)
            self.track_back.update(robot_map_loc)
        else:
            print(f'translation failed') 
            pp('robot_map_loc', robot_map_loc)

            # print(f'robot_map_loc {robot_map_loc} traversable.shape {traversable.shape}')
            print(f'is robot_loc traversable {traversable[round(robot_map_loc[1]), round(robot_map_loc[0])]}')
            print(f'is stg traversable {traversable[self.stg[1], self.stg[0]]}')

            # set map builder as obstacle 
            # TODO use robot_map_loc to update map builder instead of stg (which might be faraway and traversable)
            # self.map_builder.map[round(robot_map_loc[1]), round(robot_map_loc[0]), 1] = 1
            self.map_builder.map[self.stg[1], self.stg[0], 1] = 1
            ostg = self.stg
            # print(f'map_builder loc update {self.map_builder.map[round(robot_map_loc[1]), round(robot_map_loc[0]), 1]}')
            
            # traversable = self.update_map()
            traversable = self.get_stg(step_size)
            print(f'ostg {ostg}, stg {self.stg}')
            #check here that fmm_dist is updated 

            ob = [x for x in zip(*np.where(self.planner.fmm_dist == 10000))]
            if (ostg[1], ostg[0]) in ob:
                print(f'stg added to {len(ob)} obstaceles')
            else:
                print(f'stg not found in {len(ob)} obstacles')

            print(f'is robot_loc traversable after update {traversable[round(robot_map_loc[1]), round(robot_map_loc[0])]}')
            print(f'is stg traversable after update {traversable[ostg[1], ostg[0]]}')
            print(f'self.planner.fmm_dist[{ostg[1]}][{ostg[0]}] = {self.planner.fmm_dist[ostg[1], ostg[0]]}')
            # track back 
            track_back = self.map2real(self.track_back.get_loc(robot_map_loc, traversable))
            track_back_g = self.get_absolute_goal((track_back[0], track_back[1], 0))
            self.robot.base.go_to_absolute(track_back_g, wait=self.exec_wait)
            while self.robot.base._as.get_state() == LocalActionStatus.ACTIVE:
                pass
            if self.robot.base._as.get_state() == LocalActionStatus.SUCCEEDED:
                print(f'track back succeeded to {self.track_back}')
                robot_map_loc = self.real2map(
                    du.get_relative_state(self.get_robot_global_state(), self.init_state)
                )
                pp('robot_map_loc', robot_map_loc)
                # print(f'robot_map_loc {round(robot_map_loc[0]), round(robot_map_loc[1])}')
            else:
                print('track back failed') # possible mode of failure. shouldn't happen? check in noisy setting.

        robot_state = du.get_relative_state(self.get_robot_global_state(), self.init_state)
        # print("bot_state after executing action = {}".format(robot_state))

        # update robot location list
        robot_state_map = self.real2map(robot_state[:2])
        self.robot_loc_list_map = np.concatenate(
            (self.robot_loc_list_map, np.array([robot_state_map]))
        )
        self.prev_bot_state = robot_state
        
        # print(f'robot_state_map {robot_state_map, self.goal_loc_map, np.array(robot_state_map)}')
        print(f'distance to goal {round(np.linalg.norm(np.array(robot_state_map) - np.array(self.goal_loc_map)))}')
        # return True if robot reaches within threshold
        if (
            np.linalg.norm(np.array(robot_state_map) - np.array(self.goal_loc_map)) * 100.0
            < np.sqrt(2) * self.map_builder.resolution
        ):
            print("robot has reached goal")
            return True

        # return False if goal is not reachable
        if not traversable[round(self.goal_loc_map[1]), round(self.goal_loc_map[0])]:
            print("Goal Not reachable")
            return False
        if (
            self.planner.fmm_dist[round(robot_state_map[1]), round(robot_state_map[0])]
            >= self.planner.fmm_dist.max()
        ):
            print("whole area is explored")
            self.whole_area_explored = True
            return False
        return None
    
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

    def real2map(self, loc):
        return self.map_builder.real2map(loc)
    
    def map2real(self, loc):
        return self.map_builder.map2real(loc)

    def get_robot_global_state(self):
        """
        :return: return the global state of the robot [x_robot_loc, y_robot_loc, yaw_robot]
        :rtype: tuple
        """
        return self.robot.base.get_state("odom")
