import numpy as np
import skfmm
from numpy import ma


class FMMPlanner(object):
    def __init__(self, traversable, step_size=5):
        """

        :param traversable: 2D np.ndarray boolean map , False for obstacle and True for free, unknow space
        :param step_size: number of stapes agent suppose to travel in every short steps it takes towards goal

        :type traversable: np.ndarray
        :type step_size: int
        """
        self.step_size = step_size
        self.traversable = traversable
        self.last_goal = None

    def set_goal(self, goal):
        """
        Helps to set the goal and calculate distance from goal, try to visualize dd to get more intuition
        :param goal: goal points in map space [x_goal_co-ordinate, y_goal_co-ordinate]
        :type goal: list
        """
        traversable_ma = ma.masked_values(self.traversable * 1, 0)
        goal_x, goal_y = int(goal[0]), int(goal[1])
        traversable_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversable_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd

    def get_short_term_goal(self, state):
        """
        Given the current state of robot, function outputs where should robot move based on map and step size
        :param state: state of robot in map space [x_robot_map_co-ordinate, y_robot_map_co-ordinate]
        :type state: list
        :return: short term goal in map space where robot should move [x_map_co-ordinate, y_map_co-ordinate]
        :rtype: list
        """
        state = [int(x) for x in state]
        # pad the map with
        # to handle corners pad the dist with step size and values equal to max
        dist = np.pad(
            self.fmm_dist, self.step_size, "constant", constant_values=self.fmm_dist.shape[0] ** 2
        )
        # take subset of distance around the start, as its padded start should be corner instead of center
        subset = dist[
            state[0] : state[0] + 2 * self.step_size + 1,
            state[1] : state[1] + 2 * self.step_size + 1,
        ]

        # find the index which has minimum distance
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)
        # print(f'self.last_goal {self.last_goal}')
        if self.last_goal:
            if stg_x == self.last_goal[0] and stg_y == self.last_goal[1]:
                print('last goal was same')
                (stg_x, stg_y) = np.unravel_index(np.argpartition(subset, 2), subset.shape)
                self.last_goal = (stg_x, stg_y)
        self.last_goal = (stg_x, stg_y)
        # convert index from subset frame
        return (stg_x + state[0] - self.step_size) + 0.5, (stg_y + state[1] - self.step_size) + 0.5
