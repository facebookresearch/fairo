import os
import torch
import numpy as np
import skimage.morphology

from .visualization.semantic_exploration_vis import SemanticExplorationVisualization
from .policy.goal_policy import GoalPolicy
from droidlet.perception.robot.semantic_mapper.constants import coco_categories


class ModularSemanticScout:
    def __init__(self,
                 mover,
                 object_goal: str,
                 exploration_method: str,
                 debug=False,
                 visualize=True,
                 max_steps=400,
                 steps_per_goal=10):
        assert exploration_method in ["learned", "heuristic"]
        self.exploration_method = exploration_method
        assert (
            object_goal in coco_categories
        ), f"Object goal must be in {list(coco_categories.keys())}"
        print(
            f"Starting a semantic exploration task to reach a {object_goal} with "
            f"a modular policy and {exploration_method} exploration"
        )

        self.visualize = visualize
        self.debug = debug
        self.max_steps = max_steps
        self.steps_per_goal = steps_per_goal
        self.object_goal = object_goal
        self.object_goal_cat = coco_categories[object_goal]
        self.object_goal_cat_tensor = torch.tensor([self.object_goal_cat])
        num_sem_categories = len(coco_categories)
        self.map_size, self.local_map_size = mover.slam.get_map_sizes()

        self.goal_policy = GoalPolicy(
            map_features_shape=(
                num_sem_categories + 8,
                self.local_map_size,
                self.local_map_size
            ),
            num_outputs=2,
            hidden_size=256,
            num_sem_categories=num_sem_categories,
        )

        # FIXME Don't load this here - make an entry in artifacts and don't re-init model
        #  when this object is initialized
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
            self.vis = SemanticExplorationVisualization(self.object_goal, path=self.vis_path)

        self.step_count = 0
        self.steps_with_goal_remaining = 0
        self.finished = False
        self.goal_in_world = None

    def get_search_target_with_learned_policy(self, mover):
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
            f"Step {self.step_count}: No {self.object_goal} in the semantic map, "
            f"starting a go_to_absolute {(*goal_in_world, 0)} to find one"
        )
        if self.visualize:
            goal_map = np.zeros((self.map_size, self.map_size))
            goal_map[int(goal_in_global_map[1]), int(goal_in_global_map[0])] = 1
            self.vis.set_location_goal(goal_map)
        return goal_in_world

    def get_search_target_with_heuristic_policy(self, mover, sem_map):
        goal_map = sem_map[1, :, :] == 0

        # Set a disk around the robot to explored
        # TODO Check that the explored disk fits in the map
        radius = 10
        explored_disk = skimage.morphology.disk(radius)
        x, y = [
            int(coord) for coord in mover.slam.robot2map(mover.bot.get_base_state()[:2])
        ]
        goal_map[y - radius: y + radius + 1, x - radius: x + radius + 1][
            explored_disk == 1
            ] = 0

        # Select the frontier
        goal_map = 1 - skimage.morphology.binary_dilation(
            1 - goal_map, skimage.morphology.disk(10)
        ).astype(int)
        goal_map = (
                skimage.morphology.binary_dilation(
                    goal_map, skimage.morphology.disk(1)
                ).astype(int)
                - goal_map
        )

        if self.visualize:
            self.vis.set_location_goal(goal_map)

        return goal_map

    def step(self, mover):
        self.step_count += 1
        print("self.step_count", self.step_count)
        sem_map = mover.slam.get_global_semantic_map()
        cat_sem_map = sem_map[self.object_goal_cat + 4, :, :]

        if (cat_sem_map == 1).sum() > 0:
            # If the object goal category is present in the local map, go to it
            print(
                f"Step {self.step_count}: Found a {self.object_goal} in the semantic map, "
                f"going towards it"
            )
            goal_map = cat_sem_map == 1

            if self.visualize:
                self.vis.set_location_goal(goal_map)

            # TODO The visualization is not going to be updated during these steps
            _, goal_reached = mover.nav.go_to_absolute(goal_map=goal_map, steps=50).wait()

            if goal_reached:
                self.finished = True

        elif self.exploration_method == "learned":
            # Else if the object goal category is not present in the local map,
            # predict where to explore next with either a learned policy...
            if self.steps_with_goal_remaining == 0:
                self.goal_in_world = self.get_search_target_with_learned_policy(mover)
                self.steps_with_goal_remaining = self.steps_per_goal
            else:
                self.steps_with_goal_remaining -= 1

            mover.nav.go_to_absolute(goal=(*self.goal_in_world, 0), steps=1).wait()

        elif self.exploration_method == "heuristic":
            # ... or frontier exploration (goal = unexplored area)
            goal_map = self.get_search_target_with_heuristic_policy(mover, sem_map)
            mover.nav.go_to_absolute(goal_map=goal_map, steps=1).wait()

        # TODO Why is the semantic map visualization updated only every ~5 steps?
        if self.vis_path is not None:
            self.vis.update_semantic_frame(mover.slam.get_last_semantic_frame())
            self.vis.update_semantic_map(mover.slam.get_global_semantic_map())
            self.vis.snapshot()

        if self.step_count > self.max_steps:
            self.finished = True

    @property
    def vis_image(self):
        return self.vis.vis_image
