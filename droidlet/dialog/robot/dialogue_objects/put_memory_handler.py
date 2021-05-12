"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

from typing import Dict, Tuple, Any, Optional

from droidlet.dialog.dialogue_objects import DialogueObject


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
        # assert self.action_dict["dialogue_type"] == "PUT_MEMORY"
        # memory_type = self.action_dict["upsert"]["memory_data"]["memory_type"]
        raise NotImplementedError
