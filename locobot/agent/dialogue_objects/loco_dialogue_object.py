"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import sys
import os
from base_agent.dialogue_objects import BotCapabilities

class LocoBotCapabilities(BotCapabilities):
    """This class represents a sub-type of the Say DialogueObject above to answer
    something about the current capabilities of the bot, to the user.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        loco_response_options = [
            "I can find your jacket",
            "I can find humans",
        ]
        self.response_options.extend(loco_response_options)
