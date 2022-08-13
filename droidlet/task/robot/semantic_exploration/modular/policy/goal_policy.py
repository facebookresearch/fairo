import torch
import torch.nn as nn

from .utils.distributions import DiagGaussian
from .utils.model import Flatten, NNBase


class Goal_Oriented_Semantic_Policy(NNBase):
    def __init__(self, map_features_shape, hidden_size, num_sem_categories):
        super(Goal_Oriented_Semantic_Policy, self).__init__(False, hidden_size, hidden_size)

        self.orientation_emb = nn.Embedding(72, 8)
        self.goal_emb = nn.Embedding(num_sem_categories, 8)

        out_size = int(map_features_shape[1] / 16.0) * int(map_features_shape[2] / 16.0)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(map_features_shape[0], 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten(),
        )

        self.linear1 = nn.Linear(out_size * 32 + 8 * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)

    def forward(self, map_features, orientation, object_goal):
        map_features = self.main(map_features)
        orientation_emb = self.orientation_emb(orientation)
        goal_emb = self.goal_emb(object_goal)
        x = torch.cat((map_features, orientation_emb, goal_emb), 1)
        x = nn.ReLU()(self.linear1(x))
        x = nn.ReLU()(self.linear2(x))
        return x


class GoalPolicy(nn.Module):
    def __init__(self, map_features_shape, num_outputs, hidden_size, num_sem_categories):
        super(GoalPolicy, self).__init__()

        self.network = Goal_Oriented_Semantic_Policy(
            map_features_shape, hidden_size, num_sem_categories
        )
        self.dist = DiagGaussian(self.network.output_size, num_outputs)

    def forward(self, map_features, orientation, object_goal, deterministic=False):
        dist = self.dist(self.network(map_features, orientation, object_goal))

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # These lines
        # https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/main.py#L315
        # https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/envs/utils/fmm_planner.py#L71
        # seem to indicate that the goal action in the pre-trained model is (row, column) - i.e., we index map[goal[0], goal[1]]
        # while in this repo, this line
        # https://github.com/facebookresearch/fairo/blob/main/droidlet/lowlevel/locobot/remote/slam_pkg/utils/fmm_planner.py#L29
        # indicates that the goal action is (column, row) - i.e., we index map[goal[1], goal[0]]
        action = action.flip(-1)

        return action
