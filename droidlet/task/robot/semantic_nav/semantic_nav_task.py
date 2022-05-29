import os
import torch
import numpy as np
from droidlet.task.robot.semantic_nav.visualization.ogn_vis import (
    ObjectGoalNavigationVisualization,
)
from droidlet.task.robot.semantic_nav.policy.goal_policy import GoalPolicy

# FIXME, move this to perception
from droidlet.lowlevel.hello_robot.remote.segmentation.constants import coco_categories


# for first step of refactor, just going to rename things.
# when integrated into droidlet agent, we will merge more cleanly and make
# generalized Scout.


class ScoutObject:
    def __init__(self, mover, object_goal: str, debug=False, visualize=True, max_steps=25):
        assert (
            object_goal in coco_categories
        ), f"Object goal must be in {list(coco_categories.keys())}"
        print(f"[navigation] Starting a go_to_object {object_goal}")
        self.visualize = visualize
        self.debug = debug
        self.max_steps = max_steps
        self.object_goal = object_goal
        self.step_count = 0
        self.finished = False
        self.object_goal_cat = coco_categories[object_goal]
        self.object_goal_cat_tensor = torch.tensor([self.object_goal_cat])
        num_sem_categories = len(coco_categories)
        self.map_size, self.local_map_size = mover.slam.get_map_sizes()
        self.goal_policy = GoalPolicy(
            map_features_shape=(num_sem_categories + 8, self.local_map_size, self.local_map_size),
            num_outputs=2,
            hidden_size=256,
            num_sem_categories=num_sem_categories,
        )

        # FIXME don't load this here, properly, make an entry in artifacts,
        # don't re-init model when Scout is initted.
        policy_path = os.path.dirname(os.path.abspath(__file__)) + "/policy/goal_policy.pth"
        state_dict = torch.load(policy_path, map_location="cpu")
        self.goal_policy.load_state_dict(state_dict, strict=False)

        self.vis_path = None
        if visualize:
            try:
                # if in Habitat scene
                self.vis_path = f"images/{mover.bot.get_scene_name()}/{object_goal}"
            except:
                self.vis_path = f"images/real_world/{object_goal}"
        if self.vis_path is not None:
            self.vis = ObjectGoalNavigationVisualization(self.object_goal, path=self.vis_path)

    def get_search_target(self, mover):
        # FIXME, these are bad:
        map_features = mover.slam.get_semantic_map_features()
        orientation_tensor = mover.slam.get_orientation()

        goal_action = self.goal_policy(
            map_features, orientation_tensor, self.object_goal_cat_tensor, deterministic=False
        )[0]

        goal_in_local_map = torch.sigmoid(goal_action).numpy() * self.local_map_size
        # FIXME use droidlet coords also don't grab mover.bot directly
        global_loc = np.array(mover.slam.robot2map(mover.bot.get_base_state()[:2]))
        goal_in_global_map = global_loc + (goal_in_local_map - self.local_map_size // 2)
        goal_in_global_map = np.clip(goal_in_global_map, 0, self.map_size - 1)

        # FIXME:
        goal_in_world = mover.slam.map2robot(goal_in_global_map)

        if self.debug:
            print("goal_action:       ", goal_action)
            print("goal_in_local_map: ", goal_in_local_map)
            print("global_loc:        ", global_loc)
            print("goal_in_global_map:", goal_in_global_map)
            print("goal_in_world:     ", goal_in_world)

        print(
            f"[navigation] High-level step {self.step}: No {self.object_goal} in the semantic map, "
            f"starting a go_to_absolute {(*goal_in_world, 0)} to find one"
        )
        if self.visualize:
            goal_map = np.zeros((self.map_size, self.map_size))
            goal_map[int(goal_in_global_map[1]), int(goal_in_global_map[0])] = 1
            self.vis.add_location_goal(goal_map)
        return goal_in_world

    def step(self, mover):
        self.step_count += 1
        sem_map = mover.slam.get_global_semantic_map()
        cat_sem_map = sem_map[self.object_goal_cat + 4, :, :]

        if (cat_sem_map == 1).sum() > 0:
            # If the object goal category is present in the local map, go to it
            print(
                f"[navigation] High-level step {self.step}: Found a {self.object_goal} in the local map, "
                "starting go_to_absolute to reach it"
            )
            goal_map = cat_sem_map == 1
            if self.visualize:
                vis.add_location_goal(goal_map)
            _, goal_reached = mover.nav.go_to_absolute(goal_map=goal_map, steps=20).value
            if goal_reached:
                self.finished = True
        else:
            goal_in_world = self.get_search_target(mover)
            mover.nav.go_to_absolute(goal=(*goal_in_world, 0), steps=10)

        if self.vis_path is not None:
            self.vis.update_semantic_frame(mover.slam.get_last_semantic_frame())
            self.vis.update_semantic_map(mover.slam.get_global_semantic_map())
            self.vis.snapshot()

        if self.step_count > self.max_steps:
            self.finished = True
