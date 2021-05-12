"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import logging
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms

from .core import AbstractHandler
import droidlet.memory.robot.loco_memory as loco_memory
from droidlet.event import sio


class MemoryHandler(AbstractHandler):
    """Class for saving the state of the world parsed by all perceptual models
    to memory.

    The MemoryHandler also performs object reidentification and is responsible for maintaining
    a consistent state of the world using the output of all other perception handlers.

    Args:
        agent (LocoMCAgent): reference to the agent.
    """

    def __init__(self, agent):
        self.agent = agent
        self.init_event_handlers()

    def init_event_handlers(self):
        @sio.on("get_memory_objects")
        def objects_in_memory(sid):
            objects = loco_memory.DetectedObjectNode.get_all(self.agent.memory)
            for o in objects:
                del o["feature_repr"]
            self.agent.dashboard_memory["objects"] = objects
            sio.emit("updateState", {"memory": self.agent.dashboard_memory})

    def get_objects(self):
        return loco_memory.DetectedObjectNode.get_all(self.agent.memory)

    def handle(self, new_objects, updated_objects=[]):
        """run the memory handler for the current rgb, objects detected.

        This is also where each WorldObject is assigned a unique entity id (eid).

        Args:
            new_objects (list[WorldObject]): a list of new WorldObjects to be stored in memory
            updated_objects (list[WorldObject]): a list of WorldObjects to be updated in memory
        """
        logging.info("In MemoryHandler ... ")

        for obj in new_objects:
            obj.save_to_memory(self.agent)
        for obj in updated_objects:
            obj.save_to_memory(self.agent, update=True)
