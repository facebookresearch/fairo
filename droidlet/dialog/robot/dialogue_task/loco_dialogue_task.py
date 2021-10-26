"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import sys
import os
from droidlet.dialog.dialogue_task import BotCapabilities


class LocoBotCapabilities(BotCapabilities):
    """This class represents a sub-type of the Say DialogueObject above to answer
    something about the current capabilities of the bot, to the user.

    """

    def __init__(self, agent):
        super().__init__(agent)
        loco_response_options = ["I can find your jacket", "I can find humans"]
        self.response_options.extend(loco_response_options)
