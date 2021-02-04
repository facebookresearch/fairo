"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import sys
import os

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "../../..")
sys.path.append(BASE_AGENT_ROOT)

from base_agent.dialogue_objects import BotCapabilities


class MCBotCapabilities(BotCapabilities):
    """This class represents a sub-type of the Say DialogueObject above to answer
    something about the current capabilities of the bot, to the user.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mc_response_options = [
            'Try looking at a structure and tell me "destroy that"',
            'Try looking somewhere and tell me "build a wall there"',
            "Try building something and giving it a name",
            "Try naming something and telling me to build it",
        ]
        self.response_options.extend(mc_response_options)
