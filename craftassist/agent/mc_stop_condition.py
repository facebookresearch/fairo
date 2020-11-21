"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import sys

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_AGENT_ROOT)

from base_agent.stop_condition import StopCondition


class AgentAdjacentStopCondition(StopCondition):
    """This condition signifies if the agent is adjacent to
    a specific block type"""

    def __init__(self, agent, bid):
        super().__init__(agent)
        self.bid = bid
        self.name = "adjacent_block"

    def check(self):
        B = self.agent.get_local_blocks(1)
        return (B[:, :, :, 0] == self.bid).any()
