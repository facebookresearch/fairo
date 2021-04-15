"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from typing import Tuple, Optional

from base_agent.dialogue_objects.dialogue_object import DialogueObject


class SemanticParserWrapper(object):
    def __init__(self, agent, dialogue_object_classes, opts, dialogue_manager):
        self.dialogue_manager = dialogue_manager
        self.dialogue_objects = dialogue_object_classes
        self.agent = agent
        self.dialogue_object_parameters = {
            "agent": self.agent,
            "memory": self.agent.memory,
            "dialogue_stack": self.dialogue_manager.dialogue_stack,
        }

    def get_dialogue_object(self) -> Optional[DialogueObject]:
        raise NotImplementedError("Must implement get_dialogue_object in subclass")
