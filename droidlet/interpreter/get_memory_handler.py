"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from copy import deepcopy
import logging
from typing import Dict, Tuple, Any, Optional, Sequence

from droidlet.dialog.dialogue_task import Say
from droidlet.interpreter.interpreter import InterpreterBase
from droidlet.shared_data_structs import ErrorWithResponse
from droidlet.memory.memory_nodes import MemoryNode, TaskNode
from .interpret_filters import get_val_map

ALL_PROXIMITY = 1000


class GetMemoryHandler(InterpreterBase):
    """This class handles logical forms that ask questions about the environment or
    the assistant's current state. This requires querying the assistant's memory.

    Args:
        speaker: Name of the speaker
        logical_form_memid: pointer to memory locaiton of output of the semantic parser
        subinterpret: A dictionary that contains handlers to resolve the details of
                      salient components of a dictionary for this kind of dialogue.

    """

    def __init__(self, speaker: str, logical_form_memid: str, agent_memory, memid=None):
        super().__init__(
            speaker, logical_form_memid, agent_memory, memid=memid, interpreter_type="get_memory"
        )
        self.logical_form_orig = deepcopy(self.logical_form)
        # fill in subclasses
        self.subinterpret = {}  # noqa
        self.task_objects = {}  # noqa

    def step(self, agent) -> Tuple[Optional[str], Any]:
        """Read the action dictionary and take immediate actions based
        on memory type - either delegate to other handlers or raise an exception.

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        assert self.logical_form["dialogue_type"] == "GET_MEMORY"
        memory_type = self.logical_form["filters"].get("memory_type", "REFERENCE_OBJECT")
        if memory_type == "TASKS":
            self.handle_action(agent)
        elif memory_type == "REFERENCE_OBJECT":
            self.handle_reference_object(agent)
        else:
            raise ValueError("Unknown memory_type={}".format(memory_type))
        self.finished = True

    def handle_reference_object(self, agent, voxels_only=False) -> Tuple[Optional[str], Any]:
        """This function handles questions about a reference object and generates
        and answer based on the state of the reference object in memory.
        """
        ####TODO handle location mems too
        f = self.logical_form["filters"]

        # we should just lowercase specials in spec
        for t in f.get("triples", []):
            if t.get("obj_text") and t["obj_text"].startswith("_"):
                t["obj_text"] = t["obj_text"].lower()
        ref_obj_mems = self.subinterpret["reference_objects"](
            self,
            self.speaker,
            {"filters": f},
            extra_tags=["_not_location"],
            all_proximity=ALL_PROXIMITY,
        )
        val_map = get_val_map(self, self.speaker, f, get_all=True)
        mems, vals = val_map([m.memid for m in ref_obj_mems], [] * len(ref_obj_mems))
        # back off to tags if nothing else, FIXME do this better!
        if vals:
            if vals[0] is None:
                f["output"] = {"attribute": "tag"}
                val_map = get_val_map(self, self.speaker, f, get_all=True)
                mems, vals = val_map([m.memid for m in ref_obj_mems], [] * len(ref_obj_mems))
        self.do_answer(agent, mems, vals)

    def handle_action(self, agent) -> Tuple[Optional[str], Any]:
        """This function handles questions about the attributes and status of
        the current action.
        """
        # no clarifications etc?  FIXME:
        self.finished = True  # noqa
        f = self.logical_form["filters"]
        F = self.subinterpret["filters"](self, self.speaker, f, get_all=True)
        mems, vals = F()
        Say(agent, task_data={"response_options": str(vals)})

    def do_answer(
        self, agent, mems: Sequence[Any], vals: Sequence[Any]
    ) -> Tuple[Optional[str], Any]:
        """This function uses the action dictionary and memory state to return an answer.

        Args:
            mems: Sequence of memories
            vals: Sequence of values

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        self.finished = True  # noqa
        output_type = self.logical_form_orig["filters"].get("output")
        try:
            if type(output_type) is str and output_type.lower() == "count":
                # FIXME will multiple count if getting tags
                if not any(vals):
                    Say(agent, task_data={"response_options": "none"})
                Say(agent, task_data={"response_options": str(vals[0])})
            elif type(output_type) is dict and output_type.get("attribute"):
                attrib = output_type["attribute"]
                if type(attrib) is str and attrib.lower() == "location":
                    # add a Point task if attribute is a location
                    if self.subinterpret.get("point_target") and self.task_objects.get("point"):
                        target = self.subinterpret["point_target"].point_to_region(vals[0])
                        # FIXME agent : This is the only place in file using the agent from the .step()
                        t = self.task_objects["point"](agent, {"target": target})
                        # FIXME? higher pri, make sure this runs now...?
                        TaskNode(self.memory, t.memid)
                Say(agent, task_data={"response_options": str(vals)})
            elif type(output_type) is str and output_type.lower() == "memory":
                self.handle_exists(mems)
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

        """
        # we check the clarification because if it exists, there was a confirmation,
        # and the interpret reference object failed to find the object
        # so it does not have the proper tag.  this is an unused opportunity to learn...
        # also note if the answer is going to be no, bot will always ask.  maybe should fix this.
        clarification_query = "SELECT MEMORY FROM Task WHERE reference_object_confirmation=#={}".format(
            self.memid
        )
        _, clarification_task_mems = self.memory.basic_search(clarification_query)
        if len(mems) > 0 and len(clarification_task_mems) == 0:
            Say(agent, task_data={"response_options": "yes"})
        else:
            Say(agent, task_data={"response_options": "no"})
