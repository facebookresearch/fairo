"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from droidlet.perception.craftassist.swarm_worker_perception import SwarmLowLevelMCPerception
from agents.craftassist.craftassist_agent import CraftAssistAgent
import pdb

class CraftAssistSwarmWorker(CraftAssistAgent):
    def __init__(self, opts, idx=0):
        self.agent_idx = idx
        super(CraftAssistSwarmWorker, self).__init__(opts)

    def init_event_handlers(self):
        pass

    def init_perception(self):
        """Initialize perception modules"""
        self.perception_modules = {}
        self.perception_modules["low_level"] = SwarmLowLevelMCPerception(self)

    def init_controller(self):
        """Initialize all controllers"""
        pass

    def perceive(self, force=False):
        # disable chat modules for worker perception
        # super().perceive()
        for v in self.perception_modules.values():
            v.perceive(force=force)
