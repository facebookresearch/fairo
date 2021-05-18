"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from typing import Dict, Tuple, Any, Optional, Sequence

from . import DialogueObject, convert_location_to_selector
from droidlet.shared_data_struct.base_util import ErrorWithResponse
from droidlet.memory.memory_nodes import MemoryNode
from droidlet.interpreter.string_lists import ACTION_ING_MAPPING
from copy import deepcopy
import logging
from .filter_helper import get_val_map

ALL_PROXIMITY = 1000


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
        self.action_dict_orig = deepcopy(self.action_dict)
        # fill in subclasses
        self.subinterpret = {}  # noqa
        self.task_objects = {}  # noqa

    def step(self) -> Tuple[Optional[str], Any]:
        """Read the action dictionary and take immediate actions based
        on memory type - either delegate to other handlers or raise an exception.

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        assert self.action_dict["dialogue_type"] == "GET_MEMORY"
        memory_type = self.action_dict["filters"].get("memory_type", "REFERENCE_OBJECT")
        if memory_type == "TASKS":
            return self.handle_action()
        elif memory_type == "REFERENCE_OBJECT":
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

        # we should just lowercase specials in spec
        for t in f.get("triples", []):
            if t.get("obj_text") and t["obj_text"].startswith("_"):
                t["obj_text"] = t["obj_text"].lower()
        ref_obj_mems = self.subinterpret["reference_objects"](
            self,
            self.speaker_name,
            {"filters": f},
            extra_tags=["_not_location"],
            all_proximity=ALL_PROXIMITY,
        )
        val_map = get_val_map(self, self.speaker_name, f, get_all=True)
        mems, vals = val_map([m.memid for m in ref_obj_mems], [] * len(ref_obj_mems))
        # back off to tags if nothing else, FIXME do this better!
        if vals:
            if vals[0] is None:
                f["output"] = {"attribute": "tag"}
                val_map = get_val_map(self, self.speaker_name, f, get_all=True)
                mems, vals = val_map([m.memid for m in ref_obj_mems], [] * len(ref_obj_mems))
        return self.do_answer(mems, vals)

    def handle_action(self) -> Tuple[Optional[str], Any]:
        """This function handles questions about the attributes and status of
        the current action.

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        # no clarifications etc?  FIXME:
        self.finished = True  # noqa
        f = self.action_dict["filters"]
        F = self.subinterpret["filters"](self, self.speaker_name, f, get_all=True)
        mems, vals = F()
        return str(vals), None

    def do_answer(self, mems: Sequence[Any], vals: Sequence[Any]) -> Tuple[Optional[str], Any]:
        """This function uses the action dictionary and memory state to return an answer.

        Args:
            mems: Sequence of memories
            vals: Sequence of values

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        self.finished = True  # noqa
        output_type = self.action_dict_orig["filters"].get("output")
        try:
            if type(output_type) is str and output_type.lower() == "count":
                # FIXME will multiple count if getting tags
                if not any(vals):
                    return "none", None
                return str(vals[0]), None
            elif type(output_type) is dict and output_type.get("attribute"):
                attrib = output_type["attribute"]
                if type(attrib) is str and attrib.lower() == "location":
                    # add a Point task if attribute is a location
                    target = self.subinterpret["point_target"].point_to_region(vals[0])
                    t = self.task_objects["point"](self.agent, {"target": target})
                    self.append_new_task(t)
                return str(vals[0]), None
            elif type(output_type) is str and output_type.lower() == "memory":
                return self.handle_exists(mems)
            else:
                raise ValueError("Bad answer_type={}".format(output_type))
        except IndexError:  # index error indicates no answer available
            logging.error("No answer available from do_answer")
            raise ErrorWithResponse("I don't understand what you're asking")
        except Exception as e:
            logging.exception(e)

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
