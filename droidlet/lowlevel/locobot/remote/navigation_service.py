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
from droidlet.lowlevel.pyro_utils import safe_call
import skimage.morphology
import cv2

from slam_pkg.utils import depth_util as du
from visualization.ogn_vis import ObjectGoalNavigationVisualization
from policy.goal_policy import GoalPolicy
from policy.active_learning_policy import ActiveLearningPolicy
from segmentation.constants import coco_categories

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4


def draw_line(start, end, mat, steps=25, w=1):
    max_r, max_c = mat.shape
    start = (np.clip(start[0], 0, max_r), np.clip(start[1], 0, max_c))
    end = (np.clip(end[0], 0, max_r), np.clip(end[1], 0, max_c))
    for i in range(steps + 1):
        r = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        c = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[r - w : r + w, c - w : c + w] = 1
    return mat


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
        # if ans is None:
        #     print("couldn't find a trackback location")
        #     # if no trackback option found, then
        #     # try to trackback to a random location within a neighborhood
        #     # around current location
        #     x = cur_loc[0] + random.uniform(-0.2, 0.2)
        #     y = cur_loc[1] + random.uniform(-0.2, 0.2)
        #     angle = random.uniform(-math.pi, math.pi)
        #     ans = (x, y, angle)
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

        # ObjectNav policy
        self.goal_policy = GoalPolicy(
            map_features_shape=(num_sem_categories + 8, self.local_map_size, self.local_map_size),
            num_outputs=2,
            hidden_size=256,
            num_sem_categories=num_sem_categories,
        )
        state_dict = torch.load("policy/goal_policy.pth", map_location="cpu")
        self.goal_policy.load_state_dict(state_dict, strict=False)

        # Active learning - Learned exploration policy
        self.active_learning_learned_policy = ActiveLearningPolicy(
            map_features_shape=(num_sem_categories + 8, self.local_map_size, self.local_map_size),
            num_outputs=2,
            hidden_size=256,
        )
        state_dict = torch.load(
            "policy/active_learning_policies/active_learning_learned_policy.pth",
            map_location="cpu",
        )
        self.active_learning_learned_policy.load_state_dict(state_dict, strict=False)

        # Active learning - Learned SEAL policy
        self.active_learning_seal_policy = ActiveLearningPolicy(
            map_features_shape=(num_sem_categories + 8, self.local_map_size, self.local_map_size),
            num_outputs=2,
            hidden_size=256,
        )
        state_dict = torch.load(
            "policy/active_learning_policies/active_learning_seal_policy.pth", map_location="cpu"
        )
        self.active_learning_seal_policy.load_state_dict(state_dict, strict=False)

        self._busy = False
        self._stop = True
        self._done_exploring = False

        self.vis = ObjectGoalNavigationVisualization()

    def go_to_relative(self, goal, distance_threshold=None, angle_threshold=None):
        robot_loc = self.robot.get_base_state()
        abs_goal = du.get_relative_state(goal, (0.0, 0.0, -robot_loc[2]))
        abs_goal = list(abs_goal)
        abs_goal[0] += robot_loc[0]
        abs_goal[1] += robot_loc[1]
        abs_goal[2] = goal[2] + robot_loc[2]
        return self.go_to_absolute(
            goal=abs_goal, distance_threshold=distance_threshold, angle_threshold=angle_threshold
        )

    def execute_low_level_command(self, action, forward_dist, turn_angle, tilt_angle):
        """
        This function only works on the robot, we write it here because we want to use
        the trackback logic and we'll throw away this code.
        """

        def trackback():
            robot_loc = self.robot.get_base_state()
            trackback_loc = self.trackback.get_loc(robot_loc)
            if trackback_loc is not None:
                print(f"Tracking back to {trackback_loc}")
                trackback_status, _ = self.robot.go_to_absolute(trackback_loc, trackback=True)
                print(f"Trackback status: {trackback_status}")
            else:
                print("Could not find a trackback location. Staying in place")

        if action == 1:
            print("Starting forward action")
            is_obstacle = self.robot.is_obstacle_in_front()
            if is_obstacle:
                print("Found obstacle before translating. Aborting and tracking back")
                trackback()
                return "FAILED"
            self.robot.translate_by(forward_dist)
            self.robot.push_command()
            time.sleep(2)
            self.robot.pull_status()
            is_moving = True
            while is_moving:
                is_obstacle = self.robot.is_obstacle_in_front()
                if is_obstacle:
                    print("Found obstacle while translating. Aborting and tracking back")
                    self.robot.stop()
                    trackback()
                    return "FAILED"
                time.sleep(0.1)
                self.robot.pull_status()
                is_moving = self.robot.is_base_moving()
            print("Successful forward action")

            # Successful forward action => add new trackback loc
            robot_loc = self.robot.get_base_state()
            self.trackback.update(robot_loc)
            print("Added new trackback loc")

        elif action == 2:
            print("Starting left action")
            self.robot.rotate_by(turn_angle)
            self.robot.push_command()
            time.sleep(1)
            is_moving = True
            while is_moving:
                time.sleep(0.1)
                self.robot.pull_status()
                is_moving = self.robot.is_base_moving()
            print("Successful left action")

        elif action == 3:
            print("Starting right action")
            self.robot.rotate_by(-turn_angle)
            self.robot.push_command()
            time.sleep(1)
            is_moving = True
            while is_moving:
                time.sleep(0.1)
                self.robot.pull_status()
                is_moving = self.robot.is_base_moving()
            print("Successful right action")

        elif action == 4:
            print("Starting look up action")
            current_tilt = self.robot.get_tilt()
            self.robot.set_tilt(current_tilt + tilt_angle)

        elif action == 5:
            print("Starting look down action")
            current_tilt = self.robot.get_tilt()
            self.robot.set_tilt(current_tilt - tilt_angle)

        return "SUCCEEDED"

    def go_to_absolute(
        self,
        goal=None,
        goal_map=None,
        distance_threshold=None,
        angle_threshold=None,
        steps=100000000,
        visualize=True,
    ):
        print("[navigation] Starting a go_to_absolute")
        # specify exactly one of goal or goal_map
        assert (goal is not None and goal_map is None) or (goal is None and goal_map is not None)

        self._busy = True
        self._stop = False
        robot_loc = self.robot.get_base_state()
        initial_robot_loc = robot_loc
        goal_reached = False
        path_found = True

        while not goal_reached and steps > 0 and self._stop is False:
            stg = self.planner.get_short_term_goal(
                robot_loc,
                goal=goal,
                goal_map=goal_map,
                vis_path=f"{self.vis.path}/planner/step{self.vis.snapshot_idx}.png",
            )
            if stg == False:
                # no path to end-goal
                print(
                    "Could not find a path to the end goal {} from current robot location {}, aborting move".format(
                        goal, robot_loc
                    )
                )
                path_found = False
                break
            robot_loc = self.robot.get_base_state()
            status, action = self.robot.go_to_absolute(stg)
            robot_loc = self.robot.get_base_state()

            print("[navigation] Finished a go_to_absolute")
            print(
                " Initial location: {} Final goal: {}".format(
                    initial_robot_loc, goal if goal is not None else "goal map"
                )
            )
            print(" Short-term goal: {}, Reached Location: {}".format(stg, robot_loc))
            print(" Robot Status: {}".format(status))
            if status == "SUCCEEDED":
                goal_reached = self.planner.goal_within_threshold(
                    robot_loc,
                    goal=goal,
                    goal_map=goal_map,
                    distance_threshold=distance_threshold,
                    angle_threshold=angle_threshold,
                )
                self.trackback.update(robot_loc)
            else:
                # collided with something unexpected
                robot_loc = self.robot.get_base_state()

                # add an obstacle where the collision occurred
                print(f" Collided at {robot_loc}. Adding an obstacle to the map")
                # Robot settings
                # width = 3  # width of obstacle rectangle
                # length = 2  # depth of obstacle rectangle
                # buf = 1  # buffer space between robot and obstacle placed in front of it

                # Habitat settings
                width = 7  # width of obstacle rectangle
                length = 4  # depth of obstacle rectangle
                buf = 1  # buffer space between robot and obstacle placed in front of it

                x1, y1, t1 = robot_loc
                obstacle_locs = []
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * ((i + buf) * np.cos(t1) + (j - width // 2) * np.sin(t1))
                        wy = y1 + 0.05 * ((i + buf) * np.sin(t1) - (j - width // 2) * np.cos(t1))
                        obstacle_locs.append((wx, wy))

                self.slam.add_obstacles(obstacle_locs)

                # trackback to a known good location
                trackback_loc = self.trackback.get_loc(robot_loc)
                if trackback_loc is not None:
                    print(f"Tracking back to {trackback_loc}")
                    trackback_status, _ = self.robot.go_to_absolute(trackback_loc, trackback=True)
                    print(" Robot Trackback Status: {}".format(trackback_status))

                else:
                    print("Could not find a track back location. Staying in place")

                # TODO: if the trackback fails, we're screwed. Handle this robustly.

            steps = steps - 1

            if visualize:
                self.vis.set_action_and_collision(
                    {"action": action, "collision": status != "SUCCEEDED"}
                )
                self.vis.update_last_position_vis_info(self.slam.get_last_position_vis_info())
                self.vis.snapshot()

        self._busy = False
        return path_found, goal_reached

    def go_to_object(
        self,
        object_goal: str,
        episode_id: str,
        exploration_method="learned",
        debug=False,
        visualize=True,
        max_steps=400,
        start_with_panorama=True,
    ):
        assert exploration_method in ["learned", "frontier"]
        assert (
            object_goal in coco_categories
        ), f"Object goal must be in {list(coco_categories.keys())}"
        print(
            f"[navigation] Starting a go_to_object {object_goal} with "
            f"{exploration_method} exploration"
        )

        if visualize:
            subpath = "modular_learned" if exploration_method == "learned" else "modular_frontier"
            vis_path = f"trajectories/{episode_id}/{subpath}"
            self.vis = ObjectGoalNavigationVisualization(object_goal, path=vis_path)

        object_goal_cat = coco_categories[object_goal]
        object_goal_cat_tensor = torch.tensor([object_goal_cat])

        goal_reached = False
        high_level_step = 0
        low_level_step = 0
        low_level_steps_with_goal_remaining = 0
        if start_with_panorama:
            panorama_yaws = np.arange(0.5, 6.5, 0.5)[::-1]
            panorama_steps_remaining = len(panorama_yaws)
        else:
            panorama_steps_remaining = 0

        while not goal_reached and low_level_step < max_steps:
            low_level_step += 1

            info = self.slam.get_last_position_vis_info()
            sem_map = info["semantic_map"]
            sem_frame = info["unfiltered_semantic_frame"]
            pose = info["pose"]

            cat_sem_map = sem_map[object_goal_cat + 4, :, :]
            cat_frame = sem_frame[:, :, object_goal_cat]

            if (cat_sem_map == 1).sum() > 0:
                # If the object goal category is present in the local map, go to it
                high_level_step += 1
                panorama_steps_remaining = 0
                print(
                    f"[navigation] High-level step {high_level_step}: Found a {object_goal} in the local map, "
                    "starting go_to_absolute to reach it"
                )
                goal_map = cat_sem_map == 1

                if visualize:
                    self.vis.set_location_goal(goal_map)

                _, goal_reached = self.go_to_absolute(
                    goal_map=goal_map,
                    distance_threshold=0.5,
                    angle_threshold=30,
                    steps=1,
                    visualize=visualize,
                )
                continue

            elif (cat_frame == 1).sum() > 0:
                # Else if an instance of the object goal category is detected in
                # the frame, go in its direction
                high_level_step += 1
                low_level_steps_with_goal_remaining = 20
                panorama_steps_remaining = 0
                print(
                    f"[navigation] High-level step {high_level_step}: Found a {object_goal} in the frame, "
                    "starting go_to_absolute in its direction"
                )

                # Select unexplored area
                frontier_map = sem_map[1, :, :] == 0

                # Dilate explored area
                frontier_map = 1 - skimage.morphology.binary_dilation(
                    1 - frontier_map, skimage.morphology.disk(10)
                ).astype(int)

                # Select the frontier
                frontier_map = (
                    skimage.morphology.binary_dilation(
                        frontier_map, skimage.morphology.disk(1)
                    ).astype(int)
                    - frontier_map
                )

                # Select the intersection between the frontier and the
                # direction of the object beyond the maximum depth

                # TODO Make this adaptive
                map_size = 480
                hfov = 42
                frame_width = 480

                start_y, start_x, agent_angle = *self.slam.robot2map(pose[:2]), -pose[2]
                line_length = map_size
                median_col = np.median(np.nonzero(cat_frame)[1])
                frame_angle = np.deg2rad(median_col / frame_width * hfov - hfov / 2)
                angle = agent_angle + frame_angle

                end_y = start_y + line_length * math.sin(angle)
                end_x = start_x + line_length * math.cos(angle)
                direction_map = np.zeros((map_size, map_size))
                draw_line((start_x, start_y), (end_x, end_y), direction_map, steps=line_length)
                direction_frontier_map = frontier_map * direction_map
                goal_map = direction_frontier_map

                if debug:
                    print()
                    print(f"Step {low_level_step}")
                    print("start_x, start_y", (start_x, start_y))
                    print("end_x, end_y", (end_x, end_y))
                    print("agent_angle", np.rad2deg(agent_angle))
                    print("frame_angle", np.rad2deg(frame_angle))
                    print("angle", np.rad2deg(angle))
                    print()
                    os.makedirs("debug", exist_ok=True)
                    cv2.imwrite(
                        f"debug/cat_frame_{low_level_step}.png",
                        (cat_frame * 255).astype(np.uint8),
                    )
                    cv2.imwrite(
                        f"debug/direction_map_{low_level_step}.png",
                        (direction_map.T * 255).astype(np.uint8),
                    )
                    cv2.imwrite(
                        f"debug/frontier_map_{low_level_step}.png",
                        (frontier_map.T * 255).astype(np.uint8),
                    )
                    cv2.imwrite(
                        f"debug/direction_frontier_map_{low_level_step}.png",
                        (direction_frontier_map.T * 255).astype(np.uint8),
                    )

                if visualize:
                    self.vis.set_location_goal(goal_map)

            elif panorama_steps_remaining > 0:
                # Else if we're starting with a panorama and it's not done yet, take
                # the next step
                high_level_step += 1
                print(
                    f"[navigation] High-level step {high_level_step}: "
                    f"{panorama_steps_remaining} panorama steps remaining"
                )
                panorama_steps_remaining -= 1
                yaw = panorama_yaws[panorama_steps_remaining]
                self.go_to_absolute(goal=(0, 0, yaw), distance_threshold=0.5, angle_threshold=30)

                # In Habitat, the agent turns too fast to update the semantic map
                time.sleep(1)
                continue

            elif exploration_method == "learned":
                # Else if the object goal category is not present in the local map,
                # predict where to explore next with either a learned policy...
                if low_level_steps_with_goal_remaining == 0:
                    high_level_step += 1
                    low_level_steps_with_goal_remaining = 10

                    map_features = self.slam.get_semantic_map_features()
                    orientation_tensor = self.slam.get_orientation()

                    goal_action = self.goal_policy(
                        map_features,
                        orientation_tensor,
                        object_goal_cat_tensor,
                        deterministic=False,
                    )[0]

                    goal_in_local_map = torch.sigmoid(goal_action).numpy() * self.local_map_size
                    global_loc = np.array(self.slam.robot2map(self.robot.get_base_state()[:2]))
                    goal_in_global_map = global_loc + (
                        goal_in_local_map - self.local_map_size // 2
                    )
                    goal_in_global_map = np.clip(goal_in_global_map, 0, self.map_size - 1)
                    goal_in_world = self.slam.map2robot(goal_in_global_map)
                    goal_map = np.zeros((self.map_size, self.map_size))
                    goal_map[int(goal_in_global_map[1]), int(goal_in_global_map[0])] = 1

                    if debug:
                        print("goal_action:       ", goal_action)
                        print("goal_in_local_map: ", goal_in_local_map)
                        print("global_loc:        ", global_loc)
                        print("goal_in_global_map:", goal_in_global_map)
                        print("goal_in_world:     ", goal_in_world)

                    print(
                        f"[navigation] High-level step {high_level_step}: No {object_goal} in the semantic map, "
                        f"starting a go_to_absolute predicted by the learned policy to find one"
                    )

                    if visualize:
                        self.vis.set_location_goal(goal_map)

                else:
                    low_level_steps_with_goal_remaining -= 1

            elif exploration_method == "frontier":
                # ... or frontier exploration (goal = unexplored area)
                if low_level_steps_with_goal_remaining == 0:
                    high_level_step += 1
                    low_level_steps_with_goal_remaining = 0

                    print(
                        f"[navigation] High-level step {high_level_step}: No {object_goal} in the semantic map, "
                        f"starting a go_to_absolute decided by frontier exploration to find one"
                    )

                    # Select unexplored area
                    goal_map = sem_map[1, :, :] == 0

                    # Dilate explored area
                    goal_map = 1 - skimage.morphology.binary_dilation(
                        1 - goal_map, skimage.morphology.disk(10)
                    ).astype(int)

                    # Select the frontier
                    goal_map = (
                        skimage.morphology.binary_dilation(
                            goal_map, skimage.morphology.disk(1)
                        ).astype(int)
                        - goal_map
                    )

                    if visualize:
                        self.vis.set_location_goal(goal_map)

                else:
                    low_level_steps_with_goal_remaining -= 1

            self.go_to_absolute(
                goal_map=goal_map,
                distance_threshold=0.5,
                angle_threshold=30,
                steps=1,
                visualize=visualize,
            )

        self.vis.record_aggregate_metrics(last_pose=self.robot.get_base_state())

        print(f"[navigation] Finished a go_to_object {object_goal}")
        print(f"goal reached: {goal_reached}")

    def collect_data(
        self,
        episode_id: str,
        exploration_method="learned",
        debug=False,
        visualize=True,
        max_steps=400,
    ):
        assert exploration_method in ["learned", "frontier", "seal"]
        print(f"[navigation] Starting collecting data with " f"{exploration_method} exploration")

        if visualize:
            subpath = "modular_{}".format(exploration_method)
            vis_path = f"trajectories/{episode_id}/{subpath}"
            self.vis = ObjectGoalNavigationVisualization(path=vis_path)

        step = 0
        while step < max_steps:
            step += 1
            info = self.slam.get_last_position_vis_info()

            if exploration_method == "learned" or exploration_method == "seal":
                print(
                    f"[navigation] Step {step}: "
                    f"starting a go_to_absolute decided by learned policy"
                )

                # Only difference between "learned" and "seal" is in the policy weights (same model trained with different
                # reward functions).
                if exploration_method == "learned":
                    policy = self.active_learning_learned_policy
                elif exploration_method == "seal":
                    policy = self.active_learning_seal_policy

                map_features = self.slam.get_semantic_map_features()
                orientation_tensor = self.slam.get_orientation()

                goal_action = policy(
                    map_features,
                    orientation_tensor,
                    deterministic=False,
                )[0]

                goal_in_local_map = torch.sigmoid(goal_action).numpy() * self.local_map_size
                global_loc = np.array(self.slam.robot2map(self.robot.get_base_state()[:2]))
                goal_in_global_map = global_loc + (goal_in_local_map - self.local_map_size // 2)
                goal_in_global_map = np.clip(goal_in_global_map, 0, self.map_size - 1)
                goal_in_world = self.slam.map2robot(goal_in_global_map)
                goal_map = np.zeros((self.map_size, self.map_size))
                goal_map[int(goal_in_global_map[1]), int(goal_in_global_map[0])] = 1

                if debug:
                    print("goal_action:       ", goal_action)
                    print("goal_in_local_map: ", goal_in_local_map)
                    print("global_loc:        ", global_loc)
                    print("goal_in_global_map:", goal_in_global_map)
                    print("goal_in_world:     ", goal_in_world)

            elif exploration_method == "frontier":
                print(
                    f"[navigation] Step {step}: "
                    f"starting a go_to_absolute decided by frontier exploration"
                )

                # Select unexplored area
                sem_map = info["semantic_map"]
                goal_map = sem_map[1, :, :] == 0

                # Dilate explored area
                goal_map = 1 - skimage.morphology.binary_dilation(
                    1 - goal_map, skimage.morphology.disk(10)
                ).astype(int)

                # Select the frontier
                goal_map = (
                    skimage.morphology.binary_dilation(
                        goal_map, skimage.morphology.disk(1)
                    ).astype(int)
                    - goal_map
                )

            if visualize:
                self.vis.set_location_goal(goal_map)

            self.go_to_absolute(
                goal_map=goal_map,
                distance_threshold=0.5,
                angle_threshold=30,
                steps=1,
                visualize=visualize,
            )

        self.vis.record_aggregate_metrics(last_pose=self.robot.get_base_state())

        print(f"[navigation] Finished data collection.")

    def get_last_semantic_map_vis(self):
        return self.vis.vis_image

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
    with Pyro4.locateNS(host=robot_ip) as ns:
        ns.register("navigation", obj_uri)

    print("Navigation Server is started...")
    daemon.requestLoop()
