"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from typing import Dict, Tuple, Any, Optional, Sequence

from base_agent.dialogue_objects import (
    DialogueObject,
    FilterInterpreter,
    ReferenceObjectInterpreter,
    interpret_reference_object,
    ReferenceLocationInterpreter,
)
from .attribute_helper import MCAttributeInterpreter
from .spatial_reasoning import ComputeLocations
from base_agent.base_util import ErrorWithResponse
from base_agent.memory_nodes import MemoryNode
from string_lists import ACTION_ING_MAPPING
from tasks import Build
from ttad.generation_dialogues.generate_utils import prepend_a_an


class GetMemoryHandler(DialogueObject):
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
        super().__init__(**kwargs)
        self.provisional: Dict = {}
        self.speaker_name = speaker_name
        self.action_dict = action_dict
        self.subinterpret = {
            "filters": FilterInterpreter(),
            "reference_objects": ReferenceObjectInterpreter(interpret_reference_object),
            "reference_locations": ReferenceLocationInterpreter(),
            "specify_locations": ComputeLocations(),
        }
        self.subinterpret["attribute"] = MCAttributeInterpreter()

    def step(self) -> Tuple[Optional[str], Any]:
        """Read the action dictionary and take immediate actions based
        on memory type - either delegate to other handlers or raise an exception.
        
        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        assert self.action_dict["dialogue_type"] == "GET_MEMORY"
        memory_type = self.action_dict["filters"]["memory_type"]
        if type(memory_type) is dict:
            return self.handle_action()
        elif memory_type == "AGENT" or memory_type == "REFERENCE_OBJECT":
            return self.handle_reference_object()
        else:
            raise ValueError("Unknown memory_type={}".format(memory_type))
        self.finished = True

    def handle_reference_object(self, voxels_only=False) -> Tuple[Optional[str], Any]:
        """This function handles questions about a reference object and generates 
        and answer based on the state of the reference object in memory.
        
        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        ####TODO handle location mems too
        f = self.action_dict["filters"]
        memory_type = f["memory_type"]
        if memory_type != "AGENT":
            f["has_special_tag"] = "_not_location"
            F = self.subinterpret["filters"](self, self.speaker_name, f, get_all=True)
            mems, vals = F()
            # back off to tags if nothing else, FIXME do this better!
            if vals:
                if vals[0] is None:
                    f["output"] = {"attribute": "has_tag"}
                    F = self.subinterpret["filters"](self, self.speaker_name, f, get_all=True)
                    mems, vals = F()

        else:  # FIXME fix filters spec to not need doing agent in special case
            attribute_d = f["output"].get("attribute")
            if not attribute_d:
                raise ErrorWithResponse("output about agent is not attribute {}".format(f))
            A = self.subinterpret["attribute"](self, self.speaker_name, f["output"]["attribute"])
            mems = [self.memory.get_mem_by_id(self.memory.self_memid)]
            vals = A(mems)
        # for now, grab the first mem only, FIXME!!!!
        return self.do_answer(mems, vals)

    def handle_action(self) -> Tuple[Optional[str], Any]:
        """This function handles questions about the attributes and status of 
        the current action.
        
        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        # no clarifications etc?  FIXME:
        self.finished = True
        # get current action
        target_action_type = (
            self.action_dict["filters"].get("memory_type", {}).get("action_type", "NULL")
        )
        if target_action_type != "NULL":
            target_action_type = target_action_type[0].upper() + target_action_type[1:].lower()
            task = self.memory.task_stack_find_lowest_instance(target_action_type)
        else:
            task = self.memory.task_stack_peek()
            if task is not None:
                task = task.get_root_task()
        if task is None:
            return "I am not doing anything right now", None

        output_type = self.action_dict["filters"].get("output")
        if type(output_type) is dict and output_type.get("attribute"):
            attribute = output_type["attribute"]
            if type(attribute) is not str:
                raise ErrorWithResponse("trying get attribute {} from action".format(attribute))
            attribute = attribute.lower()
            if attribute == "action_name":
                return "I am {}".format(ACTION_ING_MAPPING[task.action_name.lower()]), None
            elif attribute == "action_reference_object_name":
                assert isinstance(task.task, Build), task.task
                for pred, val in task.task.schematic_tags:
                    if pred == "has_name":
                        return "I am building " + prepend_a_an(val), None
                return "I am building something that is {}".format(val), None
            elif attribute == "move_target":
                assert task.action_name == "Move", task.action_name
                target = tuple(task.task.target)
                return "I am going to {}".format(target), None

    def do_answer(self, mems: Sequence[Any], vals: Sequence[Any]) -> Tuple[Optional[str], Any]:
        """This function uses the action dictionary and memory state to return an answer. 

        Args:
            mems: Sequence of memories 
            vals: Sequence of values

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        self.finished = True
        output_type = self.action_dict["filters"].get("output")
        try:
            if type(output_type) is str and output_type.lower() == "COUNT":
                # FIXME will multiple count if getting tags
                return str(len(mems)), None
            elif type(output_type) is dict and output_type.get("attribute"):
                return str(vals[0]), None
            elif type(output_type) is str and output_type.lower() == "memory":
                return self.handle_exists(mems)
            else:
                raise ValueError("Bad answer_type={}".format(output_type))
        except:
            raise ErrorWithResponse("I don't understand what you're asking")

    def handle_exists(self, mems: Sequence[MemoryNode]) -> Tuple[Optional[str], Any]:
        """Check if a memory exists.

        Args:
            mems: Sequence of memories
        
        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        # we check progeny data bc if it exists, there was a confirmation,
        # and the interpret reference object failed to find the object
        # so it does not have the proper tag.  this is an unused opportunity to learn...
        # also note if the answer is going to be no, bot will always ask.  maybe should fix this.
        if len(mems) > 0 and len(self.progeny_data) == 0:
            return "Yes", None
        else:
            return "No", None
