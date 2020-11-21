"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

"""Minimal agent stub only used to test CraftAssistAgent's __init__ function. It must be called "Agent" to mimic the agent.so import.

For a full-featured test agent to import in other unit tests, use FakeAgent.
"""


class Agent(object):
    def __init__(self, host, port, name):
        pass

    def send_chat(self, chat):
        pass
