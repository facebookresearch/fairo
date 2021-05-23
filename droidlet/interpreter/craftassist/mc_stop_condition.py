"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import os
import sys

from droidlet.interpreter.condition import Condition

# FIXME!!!! this is dead code
class AgentAdjacentStopCondition(Condition):
    """This condition signifies if the agent is adjacent to
    a specific block type"""

    def __init__(self, memory, bid):
        super().__init__(memory)
        self.bid = bid
        self.name = "adjacent_block"

    def check(self):
        B = self.agent.get_local_blocks(1)
        return (B[:, :, :, 0] == self.bid).any()
