"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from typing import Dict, Sequence

from base_agent.dialogue_objects import (
    FilterInterpreter,
    ReferenceObjectInterpreter,
    interpret_reference_object,
    ReferenceLocationInterpreter,
    GetMemoryHandler,
)
from .attribute_helper import MCAttributeInterpreter
from .spatial_reasoning import ComputeLocations
from base_agent.base_util import ErrorWithResponse
from tasks import Build, Point
from ttad.generation_dialogues.generate_utils import prepend_a_an


class MCGetMemoryHandler(GetMemoryHandler):
    """This class handles logical forms that ask questions about the environment or 
    the assistant's current state. This requires querying the assistant's memory.

    Args:
        provisional: A dictionary used to store information to support clarifications
        speaker_name: Name or id of the speaker
        action_dict: output of the semantic parser (also called the logical form).
        subinterpret: A dictionary that contains handlers to resolve the details of 
                      salient components of a dictionary for this kind of dialogue.

    """

    def __init__(self, speaker_name: str, action_dict: Dict, **kwargs):
        super().__init__(speaker_name, action_dict, **kwargs)
        self.subinterpret = {
            "filters": FilterInterpreter(),
            "reference_objects": ReferenceObjectInterpreter(interpret_reference_object),
            "reference_locations": ReferenceLocationInterpreter(),
            "specify_locations": ComputeLocations(),
        }
        self.subinterpret["attribute"] = MCAttributeInterpreter()
        self.task_objects = {
            "point": Point
        }

    def handle_task_refobj_string(self, task, refobj_attr):
        if refobj_attr == "name":
            assert isinstance(task.task, Build), task.task
            for pred, val in task.task.schematic_tags:
                if pred == "has_name":
                    return "I am building " + prepend_a_an(val), None
                return "I am building something that is {}".format(val), None
        elif refobj_attr == "location":
            assert task.action_name == "Move", task.action_name
            target = tuple(task.task.target)
            return "I am going to {}".format(target), None
        else:
            raise ErrorWithResponse("trying get attribute {} from action".format(refobj_attr))
        return None, None
