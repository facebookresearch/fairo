"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
from typing import Dict, Tuple, Any, Optional


from droidlet.dialog.dialogue_task import Say
from droidlet.interpreter import (
    FilterInterpreter,
    ReferenceObjectInterpreter,
    ReferenceLocationInterpreter,
    interpret_reference_object,
    InterpreterBase,
)
from .spatial_reasoning import ComputeLocations
from droidlet.memory.memory_nodes import TaskNode, SetNode, InterpreterNode
from droidlet.memory.craftassist.mc_memory_nodes import VoxelObjectNode, RewardNode
from droidlet.interpreter.craftassist.tasks import Point
from droidlet.shared_data_structs import ErrorWithResponse


class PutMemoryHandler(InterpreterBase):
    """This class handles logical forms that give input to the agent about the environment or
    about the agent itself. This requires writing to the assistant's memory.

    Args:
        speaker: Name or id of the speaker
        action_dict: output of the semantic parser (also called the logical form).
        subinterpret: A dictionary that contains handlers to resolve the details of
                      salient components of a dictionary for this kind of dialogue.
    """

    def __init__(self, speaker, logical_form_memid, agent_memory, memid=None, low_level_data=None):
        super().__init__(
            speaker, logical_form_memid, agent_memory, memid=memid, interpreter_type="put_memory"
        )
        self.get_locs_from_entity = low_level_data["get_locs_from_entity"]
        self.subinterpret = {
            "filters": FilterInterpreter(),
            "reference_objects": ReferenceObjectInterpreter(interpret_reference_object),
            "reference_locations": ReferenceLocationInterpreter(),
            "specify_locations": ComputeLocations(),
        }
        self.task_objects = {"point": Point}

    def step(self, agent) -> Tuple[Optional[str], Any]:
        """Take immediate actions based on action dictionary and
        mark the dialogueObject as finished.

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        r = self._step(agent)
        self.finished = True
        return r

    def _step(self, agent) -> Tuple[Optional[str], Any]:
        """Read the action dictionary and take immediate actions based
        on memory type - either delegate to other handlers or raise an exception.

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        memory_type = self.logical_form["upsert"]["memory_data"]["memory_type"]
        if memory_type == "REWARD":
            return self.handle_reward(agent)
        elif memory_type == "SET":
            return self.handle_set(agent)
        elif memory_type == "TRIPLE":
            return self.handle_triple(agent)
        else:
            raise NotImplementedError

    def handle_reward(self, agent) -> Tuple[Optional[str], Any]:
        """Creates a new node of memory type : RewardNode and
        returns a confirmation.

        """
        reward_value = self.logical_form["upsert"]["memory_data"]["reward_value"]
        assert reward_value in ("POSITIVE", "NEGATIVE"), self.logical_form
        RewardNode.create(self.memory, reward_value)
        if reward_value == "POSITIVE":
            r = "Thank you!"
        else:
            r = "I'll try to do better in the future."
        Say(agent, task_data={"response_options": r})

    def handle_triple(self, agent) -> Tuple[Optional[str], Any]:
        """Writes a triple of type : (subject, predicate_text, object_text)
        to the memory and returns a confirmation.

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        ref_obj_d = {"filters": self.logical_form["filters"]}
        r = self.subinterpret["reference_objects"](
            self, self.speaker, ref_obj_d, extra_tags=["_physical_object"]
        )
        if len(r) == 0:
            raise ErrorWithResponse("I don't know what you're referring to")
        mem = r[0]

        name = "it"
        triples = self.memory.get_triples(subj=mem.memid, pred_text="has_tag")
        if len(triples) > 0:
            name = triples[0][2].strip("_")

        schematic_memid = (
            self.memory.convert_block_object_to_schematic(mem.memid).memid
            if isinstance(mem, VoxelObjectNode)
            else None
        )

        for t in self.logical_form["upsert"]["memory_data"].get("triples", []):
            if t.get("pred_text") and t.get("obj_text"):
                logging.debug("Tagging {} {} {}".format(mem.memid, t["pred_text"], t["obj_text"]))
                self.memory.add_triple(
                    subj=mem.memid, pred_text=t["pred_text"], obj_text=t["obj_text"]
                )
                if schematic_memid:
                    self.memory.add_triple(
                        subj=schematic_memid, pred_text=t["pred_text"], obj_text=t["obj_text"]
                    )
            point_at_target = mem.get_point_at_target()
            # FIXME agent : This is the only place in file using the agent from the .step()
            task = self.task_objects["point"](agent, {"target": point_at_target})
            # FIXME? higher pri, make sure this runs now...?
            TaskNode(self.memory, task.memid)
            r = "OK I'm tagging this %r as %r %r " % (name, t["pred_text"], t["obj_text"])
            Say(agent, task_data={"response_options": r})
        return

    def handle_set(self, agent) -> Tuple[Optional[str], Any]:
        """ creates a set of memories

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        ref_obj_d = {"filters": self.logical_form["filters"]}
        ref_objs = self.subinterpret["reference_objects"](
            self, self.speaker, ref_obj_d, extra_tags=["_physical_object"]
        )
        if len(ref_objs) == 0:
            raise ErrorWithResponse("I don't know what you're referring to")

        triples_d = self.logical_form["upsert"]["memory_data"].get("triples")
        if len(triples_d) == 1 and triples_d[0]["pred_text"] == "has_name":
            # the set has a name; check to see if one with that name exists,
            # if so add to it, else create one with that name
            name = triples_d[0]["obj_text"]
            set_memids, _ = self.memory.basic_search(
                "SELECT MEMORY FROM Set WHERE (has_name={} OR name={})".format(name, name)
            )
            if not set_memids:
                # make a new set, and name it
                set_memid = SetNode.create(self.memory)
                self.memory.add_triple(subj=set_memid, pred_text="has_name", obj_text=name)
            else:
                # FIXME, which one
                set_memid = set_memids[0]
        else:
            # an anonymous set, assuming its new, and defined to hold the triple(s)
            set_memid = SetNode.create(self.memory)
            for t in triples_d:
                self.memory.add_triple(
                    subj=set_memid, pred_text=t["pred_text"], obj_text=t["obj_text"]
                )
        for r in ref_objs:
            self.memory.add_triple(subj=r.memid, pred_text="member_of", obj=set_memid)

        # FIXME point to the objects put in the set, otherwise explain this better
        Say(agent, task_data={"response_options": "OK made those objects into a set "})
        return
