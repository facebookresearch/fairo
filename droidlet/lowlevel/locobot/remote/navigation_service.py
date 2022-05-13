import os
import sys
import random
import math
from threading import local
import time
import torch
import numpy as np
import Pyro4
from rich import print

from slam_pkg.utils import depth_util as du
from policy.goal_policy import GoalPolicy
from segmentation.constants import coco_categories

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
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

        num_sem_categories = len(coco_categories)
        self.map_size, self.local_map_size = self.slam.get_map_sizes()
        self.goal_policy = GoalPolicy(
            map_features_shape=(num_sem_categories + 8, self.local_map_size, self.local_map_size),
            num_outputs=2,
            hidden_size=256,
            num_sem_categories=num_sem_categories,
        )
        state_dict = torch.load("policy/goal_policy.pth", map_location="cpu")
        self.goal_policy.load_state_dict(state_dict, strict=False)

        self._busy = False
        self._stop = True
        self._done_exploring = False

    def go_to_relative(self, goal):
        robot_loc = self.robot.get_base_state()
        abs_goal = du.get_relative_state(goal, (0.0, 0.0, -robot_loc[2]))
        abs_goal = list(abs_goal)
        abs_goal[0] += robot_loc[0]
        abs_goal[1] += robot_loc[1]
        abs_goal[2] = goal[2] + robot_loc[2]
        return self.go_to_absolute(abs_goal)

    def go_to_absolute(self, goal=None, goal_map=None, steps=100000000):
        # specify exactly one of goal or goal_map
        assert (goal is not None and goal_map is None) or (goal is None and goal_map is not None)

        self._busy = True
        self._stop = False
        robot_loc = self.robot.get_base_state()
        initial_robot_loc = robot_loc
        goal_reached = False
        path_found = True

        while (not goal_reached) and steps > 0 and self._stop is False:
            stg = self.planner.get_short_term_goal(robot_loc, goal=goal, goal_map=goal_map)
            if stg == False:
                # no path to end-goal
                print(
                    "Could not find a path to the end goal {} from current robot location {}, aborting move".format(
                        goal, robot_loc
                    )
                )
                path_found = False
                break
            status = self.robot.go_to_absolute(stg)
            robot_loc = self.robot.get_base_state()

            print("[navigation] Finished a go_to_absolute")
            print(
                " initial location: {} Final goal: {}".format(
                    initial_robot_loc, goal if goal is not None else "goal map"
                )
            )
            print(" short-term goal: {}, Reached Location: {}".format(stg, robot_loc))
            print(" Robot Status: {}".format(status))
            if status == "SUCCEEDED":
                goal_reached = self.planner.goal_within_threshold(
                    robot_loc, goal=goal, goal_map=goal_map
                )
                self.trackback.update(robot_loc)
            else:
                # collided with something unexpected.
                robot_loc = self.robot.get_base_state()

                # trackback to a known good location
                trackback_loc = self.trackback.get_loc(robot_loc)

                print(f"Collided at {robot_loc}." f"Tracking back to {trackback_loc}")
                self.robot.go_to_absolute(trackback_loc)
                # TODO: if the trackback fails, we're screwed. Handle this robustly.
            steps = steps - 1

        self._busy = False
        return path_found, goal_reached

    def go_to_object(self, object_goal: str, debug=True):
        assert (
            object_goal in coco_categories
        ), f"Object goal must be in {list(coco_categories.keys())}"
        print(f"[navigation] Starting a go_to_object {object_goal}")
        object_goal_cat = coco_categories[object_goal]
        object_goal_cat_tensor = torch.tensor([object_goal_cat])

        goal_reached = False

        while not goal_reached:
            sem_map = self.slam.get_global_semantic_map()
            cat_sem_map = sem_map[object_goal_cat + 4, :, :]

            if (cat_sem_map == 1).sum() > 0:
                # If the object goal category is present in the local map, go to it
                print(
                    f"[navigation] Found a {object_goal} in the local map, starting "
                    "go_to_absolute to reach it"
                )
                goal_map = cat_sem_map == 1
                _, goal_reached = self.go_to_absolute(goal_map=goal_map, steps=25)

            else:
                # Else if the object goal category is not present in the local map,
                # predict where to explore next
                map_features = self.slam.get_semantic_map_features()
                orientation_tensor = self.slam.get_orientation()

                goal_action = self.goal_policy(
                    map_features, orientation_tensor, object_goal_cat_tensor, deterministic=False
                )[0]
                # These lines
                # https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/main.py#L315
                # https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/envs/utils/fmm_planner.py#L71
                # indicate that the goal action in the pre-trained model is (row, column) - i.e., we index map[goal[0], goal[1]]
                # while in this repo, this line
                # https://github.com/facebookresearch/fairo/blob/main/droidlet/lowlevel/locobot/remote/slam_pkg/utils/fmm_planner.py#L29
                # indicates that the goal action is (column, row) - i.e., we index map[goal[1], goal[0]]
                # goal_action = goal_action.flip(0)
                goal_in_local_map = torch.sigmoid(goal_action).numpy() * self.local_map_size
                global_loc = np.array(self.slam.real2map(self.robot.get_base_state()[:2]))
                goal_in_global_map = global_loc + (goal_in_local_map - self.local_map_size // 2)
                goal_in_global_map = np.clip(goal_in_global_map, 0, self.map_size - 1)
                goal_in_world = self.slam.map2real(goal_in_global_map)

                if debug:
                    print("goal_action:       ", goal_action)
                    print("goal_in_local_map: ", goal_in_local_map)
                    print("global_loc:        ", global_loc)
                    print("goal_in_global_map:", goal_in_global_map)
                    print("goal_in_world:     ", goal_in_world)

                print(
                    f"[navigation] No {object_goal} in the semantic map, starting a "
                    f"go_to_absolute goal={(*goal_in_world, 0)} to find one"
                )
                _, goal_reached = self.go_to_absolute(goal=(*goal_in_world, 0), steps=25)

        print(f"[navigation] Finished a go_to_object {object_goal}")

    def explore(self, far_away_goal):
        if not hasattr(self, "_done_exploring"):
            self._done_exploring = False
        if not self._done_exploring:
            print("exploring 1 step")
            path_found, _ = self.go_to_absolute(far_away_goal, steps=1)
            if path_found == False:
                # couldn't reach far_away_goal
                # and don't seem to have any unexplored
                # paths to attempt to get there
                self._done_exploring = True
                print("exploration done")

    def is_busy(self):
        return self._busy

    def is_done_exploring(self):
        return self._done_exploring

    def reset_explore(self):
        self._done_exploring = False

    def stop(self):
        self._stop = True


robot_ip = os.getenv("LOCOBOT_IP")
ip = os.getenv("LOCAL_IP")

robot_name = "remotelocobot"
if len(sys.argv) > 1:
    robot_name = sys.argv[1]

with Pyro4.Daemon(ip) as daemon:
    robot = Pyro4.Proxy("PYRONAME:" + robot_name + "@" + robot_ip)
    planner = Pyro4.Proxy("PYRONAME:planner@" + robot_ip)
    slam = Pyro4.Proxy("PYRONAME:slam@" + robot_ip)

    obj = Navigation(planner, slam, robot)
    obj_uri = daemon.register(obj)
    with Pyro4.locateNS() as ns:
        ns.register("navigation", obj_uri)

    print("Navigation Server is started...")
    daemon.requestLoop()
