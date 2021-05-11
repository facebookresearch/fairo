"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from typing import Dict, Tuple, Any, Optional, Sequence

from droidlet.dialog.dialogue_objects import (
    FilterInterpreter,
    ReferenceObjectInterpreter,
    interpret_reference_object,
    ReferenceLocationInterpreter,
    AttributeInterpreter,
    GetMemoryHandler,
)
from .spatial_reasoning import ComputeLocations
from .point_target import PointTargetInterpreter
from droidlet.base_util import ErrorWithResponse
from droidlet.memory.memory_nodes import MemoryNode, ReferenceObjectNode
from droidlet.interpreter.string_lists import ACTION_ING_MAPPING
from droidlet.dialog.ttad.generation_dialogues import prepend_a_an
from copy import deepcopy
from droidlet.interpreter.robot.tasks import Point


class LocoGetMemoryHandler(GetMemoryHandler):
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
            "attribute": AttributeInterpreter(),
            "point_target": PointTargetInterpreter(),
        }
        self.task_objects = {"point": Point}

    def handle_task_refobj_string(self, task, refobj_attr):
        if refobj_attr == "name":
            for pred, val in task.task.target:
                if pred == "has_name":
                    return "I am going to the " + prepend_a_an(val), None
        elif refobj_attr == "location":
            target = tuple(task.task.target)
            return "I am going to {}".format(target), None
        else:
            raise ErrorWithResponse("trying get attribute {} from action".format(refobj_attr))
