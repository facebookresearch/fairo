import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):
    def reset(self):
        """ Resets agent at start of trajectory """

    def forward(self, image, robot_state):
        raise NotImplementedError


class ClosedLoopAgent(Agent):
    def __init__(self, policy_network, H=30):
        super().__init__()
        self._pi = policy_network
        self._t, self._cache, self._H = 0, None, H
    
    def reset(self):
        self._t, self._cache = 0, None

    def forward(self, image, robot_state):
        if self._H == 1:
            return self._pi(image, robot_state)

        index = self._t % self._H
        if index == 0:
            self._cache = self._pi(image, robot_state).detach()
        self._t += 1
        return self._cache[:,index].detach()


class RNNAgent(Agent):
    def __init__(self, policy_network, _=30):
        super().__init__()
        self._pi = policy_network
        self._memory = None
    
    def reset(self):
        self._memory = None

    def forward(self, image, robot_state):
        ac, self._memory = self._pi(image, robot_state, self._memory, True)
        return ac


class OpenLoopAgent(Agent):
    def __init__(self, policy_network):
        super().__init__()
        self._pi = policy_network
        self._t, self._cache = 0, None

    def reset(self):
        self._t, self._cache = 0, None
    
    def forward(self, image, robot_state):
        if self._t == 0:
            self._cache = self._pi(image, robot_state).detach()
        elif self._t >= self._cache.shape[1]:
            raise ValueError
        self._t += 1
        return self._cache[:,self._t-1].detach()
