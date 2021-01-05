"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import logging
from typing import Dict, Tuple, Any, Optional

from base_agent.dialogue_objects import (
    DialogueObject,
    FilterInterpreter,
    ReferenceObjectInterpreter,
    ReferenceLocationInterpreter,
    interpret_reference_object,
)
from .spatial_reasoning import ComputeLocations
from mc_memory_nodes import VoxelObjectNode, RewardNode
from base_agent.base_util import ErrorWithResponse


class PutMemoryHandler(DialogueObject):
    """This class handles logical forms that give input to the agent about the environment or 
    about the agent itself. This requires writing to the assistant's memory.

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

    def step(self) -> Tuple[Optional[str], Any]:
        """Take immediate actions based on action dictionary and
        mark the dialogueObject as finished.

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        r = self._step()
        self.finished = True
        return r

    def _step(self) -> Tuple[Optional[str], Any]:
        """Read the action dictionary and take immediate actions based
        on memory type - either delegate to other handlers or raise an exception.
        
        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        assert self.action_dict["dialogue_type"] == "PUT_MEMORY"
        memory_type = self.action_dict["upsert"]["memory_data"]["memory_type"]
        if memory_type == "REWARD":
            return self.handle_reward()
        elif memory_type == "TRIPLE":
            return self.handle_triple()
        else:
            raise NotImplementedError

    def handle_reward(self) -> Tuple[Optional[str], Any]:
        """Creates a new node of memory type : RewardNode and 
        returns a confirmation.

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        reward_value = self.action_dict["upsert"]["memory_data"]["reward_value"]
        assert reward_value in ("POSITIVE", "NEGATIVE"), self.action_dict
        RewardNode.create(self.memory, reward_value)
        if reward_value == "POSITIVE":
            return "Thank you!", None
        else:
            return "I'll try to do better in the future.", None

    def handle_triple(self) -> Tuple[Optional[str], Any]:
        """Writes a triple of type : (subject, predicate_text, object_text)
        to the memory and returns a confirmation.

        Returns:
            output_chat: An optional string for when the agent wants to send a chat
            step_data: Any other data that this step would like to send to the task
        """
        ref_obj_d = {"filters": self.action_dict["filters"]}
        r = self.subinterpret["reference_objects"](
            self, self.speaker_name, ref_obj_d, extra_tags=["_physical_object"]
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

        for t in self.action_dict["upsert"]["memory_data"].get("triples", []):
            if t.get("pred_text") and t.get("obj_text"):
                logging.info("Tagging {} {} {}".format(mem.memid, t["pred_text"], t["obj_text"]))
                self.memory.add_triple(
                    subj=mem.memid, pred_text=t["pred_text"], obj_text=t["obj_text"]
                )
                if schematic_memid:
                    self.memory.add_triple(
                        subj=schematic_memid, pred_text=t["pred_text"], obj_text=t["obj_text"]
                    )
        point_at_target = mem.get_point_at_target()
        self.agent.send_chat(
            "OK I'm tagging this %r as %r %r " % (name, t["pred_text"], t["obj_text"])
        )
        self.agent.point_at(list(point_at_target))

        return "Done!", None
