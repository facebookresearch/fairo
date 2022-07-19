import numpy as np
import skfmm
from numpy import ma
import skimage.morphology
import cv2
import os


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

    def set_goal(self, goal, vis_path=None):
        """
        Helps to set the goal and calculate distance from goal, try to visualize dd to get more intuition
        :param goal: goal points in map space [x_goal_co-ordinate, y_goal_co-ordinate]
        :type goal: list
        """
        traversable_ma = ma.masked_values(self.traversable * 1, 0)
        goal_x, goal_y = round(goal[0]), round(goal[1])
        traversable_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversable_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd

        # if vis_path is not None:
        #     goal_map = np.zeros_like(self.traversable)
        #     goal_map[goal_y - 3 : goal_y + 4, goal_x - 3 : goal_x + 4] = 1
        #     self._visualize(goal_map, vis_path)

    def set_multi_goal(self, goal_map, vis_path=None):
        """Set multiple possible target goals - this is useful for object goal navigation.
        :param goal_map: binary map that contains multiple goals (encoded as 1)
        """
        selem = skimage.morphology.disk(10)
        goal_map = skimage.morphology.binary_dilation(goal_map, selem) != True
        goal_map = 1 - goal_map * 1.0
        traversible_ma = ma.masked_values(self.traversable * 1, 0)
        traversible_ma[goal_map == 1] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd

        # if vis_path is not None:
        #     self._visualize(goal_map, vis_path)

    def _visualize(self, goal_map, vis_path):
        r, c = self.traversable.shape
        dist_vis = np.zeros((c, r * 3))
        dist_vis[:, :r] = self.traversable.T
        dist_vis[:, r : 2 * r] = goal_map.T
        dist_vis[:, 2 * r :] = (self.fmm_dist / self.fmm_dist.max()).T
        dist_vis = (dist_vis * 255.0).astype(np.uint8)

        folder = "/".join(vis_path.split("/")[:-1])
        if folder != "":
            os.makedirs(folder, exist_ok=True)
        cv2.imwrite(vis_path, dist_vis)

    def get_short_term_goal(self, state):
        """
        Given the current state of robot, function outputs where should robot move based on map and step size
        :param state: state of robot in map space [x_robot_map_co-ordinate, y_robot_map_co-ordinate]
        :type state: list
        :return: short term goal in map space where robot should move [x_map_co-ordinate, y_map_co-ordinate]
        :rtype: list
        """
        state = [round(x) for x in state]
        # print(f'get stg for state {state[1], state[0]}')
        # pad the map with
        # to handle corners pad the dist with step size and values equal to max
        dist = np.pad(
            self.fmm_dist, self.step_size, "constant", constant_values=self.fmm_dist.shape[0] ** 2
        )
        # take subset of distance around the start
        subset = dist[
            state[1] + self.step_size - self.step_size : state[1] + 2 * self.step_size + 1,
            state[0] + self.step_size - self.step_size : state[0] + 2 * self.step_size + 1,
        ]

        # print(f'subset.shape {subset.shape}')

        # find the index which has minimum distance
        np.set_printoptions(precision=3)
        (stg_y, stg_x) = np.unravel_index(np.argmin(subset), subset.shape)
        # print(stg_y, stg_x, subset[stg_y][stg_x], self.step_size)

        # convert index from subset frame (return x,y)
        sx = stg_x - self.step_size + state[0]
        sy = stg_y - self.step_size + state[1]
        # print(f'self.fmm_dist {self.fmm_dist[sy][sx], self.fmm_dist[state[1]][state[0]]}')
        return sx, sy
